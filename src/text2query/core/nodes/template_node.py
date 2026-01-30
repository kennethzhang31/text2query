from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path
import yaml
import logging
import re

from .state import WisbiState
from ..utils.models import agenerate_chat, ModelConfig
from ..template_matching.store import TemplateStore
from ..template_matching.models import Template

logger = logging.getLogger(__name__)


class TemplateNode:
    """Template node with RAG-based template retrieval
    
    This class handles:
    - Planning: Extracting components from natural language queries using LLM
    - Template matching: Finding similar templates using vector similarity search
    - Auto-population: Loading templates from YAML if store is empty
    """
    
    _PROMPT = (
        "You are a planner. Produce ONLY YAML matching the schema below under a single 'components' root.\n"
        "If a field is not inferable, use null or omit it.\n\n"
        "Allowed values:\n"
        "- base_query: [timeseries, snapshot, comparison, ranking]\n"
        "- dimensions.time: [day, month, quarter, year]\n"
        "- dimensions.org: [single, multiple, all]\n"
        "- filters.standard: [date_range, org_filter]\n"
        "- filters.business: [doc_type, threshold]\n\n"
        "YAML output (no code fences):\n"
        "components:\n"
        "  base_query: <enum|null>\n"
        "  dimensions:\n"
        "    time: <enum|null>\n"
        "    org: <enum|null>\n"
        "    region: <string|null>\n"
        "    industry: <string|null>\n"
        "  filters:\n"
        "    standard: {{date_range: <bool>, org_filter: <bool>}}\n"
        "    business: {{doc_type: <bool>, threshold: <bool>}}\n\n"
        "Query: {query}\n"
    ).strip()
    
    def __init__(
        self,
        template_store: TemplateStore,
        templates_file: Optional[str] = None,
    ):
        """Initialize TemplateNode
        
        Args:
            template_store: Pre-initialized TemplateStore instance (required)
            templates_file: Path to templates.yaml file (optional, will try to find it)
        """
        if template_store is None:
            raise ValueError("template_store is required")
        self.template_store = template_store
        self.templates_file = templates_file or self._find_templates_file()

    def _find_templates_file(self) -> str:
        """Try to find templates.yaml file in common locations"""
        current_file = Path(__file__)
        possible_paths = [
            current_file.parent.parent / "utils" / "templates" / "templates.yaml",
            Path("core/utils/templates/templates.yaml"),
            Path("templates/templates.yaml"),
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
        raise FileNotFoundError("Could not find templates.yaml file")

    def _load_templates_from_yaml(self) -> Dict[str, Template]:
        """Load templates from YAML file
        
        Returns:
            Dict[str, Template]: Dictionary mapping template IDs to Template objects
        """
        try:
            with open(self.templates_file, 'r') as file:
                data = yaml.safe_load(file)
            templates = {}
            template_data = data.get('templates', {})
            for template_id, template_info in template_data.items():
                template = Template(
                    id=template_id,
                    description=template_info.get('description', ''),
                    base_query=template_info.get('base_query', None),
                    dimensions=template_info.get('dimensions', {}),
                    filters=template_info.get('filters', {}),
                    metrics=template_info.get('metrics', []),
                    sql=template_info.get('sql', None),
                )
                templates[template_id] = template
            logger.info(f"Loaded {len(templates)} templates from {self.templates_file}")
            return templates
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
            raise

    async def _populate_if_empty(self, embedder_config: ModelConfig) -> None:
        """Auto-populate templates from YAML if store is empty"""
        if await self.template_store.is_empty():
            logger.info("Template store is empty, auto-populating from templates.yaml")
            templates = self._load_templates_from_yaml()
            
            # Insert all templates in the store with SQL from YAML
            for template in templates.values():
                await self.template_store.insert_template(
                    template=template,
                    embedder_config=embedder_config,
                    sql_command=template.sql,
                )
            
            logger.info(f"Auto-populated {len(templates)} templates to store")
        else:
            logger.info("Template store already contains templates, skipping auto-population")
    
    async def _find_top_templates(
        self,
        components: Dict[str, Any],
        embedder_config: ModelConfig,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> List[Tuple[float, Template]]:
        """Find top K matching templates using RAG-based vector similarity.
        
        Args:
            components: Dictionary with base_query, dimensions, filters, etc.
            embedder_config: Embedder config for generating embeddings
            top_k: Number of top templates to return (default: 3)
            min_similarity: Minimum similarity score threshold (default: 0.0)
            
        Returns:
            List[Tuple[float, Template]]: List of (similarity_score, template) tuples,
                sorted by similarity (highest first). The template.sql field
                contains the corresponding SQL command (if available).
        """
        await self._populate_if_empty(embedder_config)
        return await self.template_store.find_similar_templates(
            components=components,
            embedder_config=embedder_config,
            top_k=top_k,
            min_similarity=min_similarity,
        )
    
    async def _plan(self, state: WisbiState, query_text: str) -> Dict[str, Any]:
        """Extract components from natural language query using LLM
        
        Args:
            state: Workflow state containing llm_config
            query_text: Natural language query
            
        Returns:
            Dict containing extracted components
        """
        prompt = self._PROMPT.format(query=query_text)
        llm_config = state.get("llm_config")
        messages = [{"role": "user", "content": prompt}]
        response = await agenerate_chat(llm_config, messages)
        sanitized = self._sanitize_yaml_response(response)
        parsed = yaml.safe_load(sanitized) or {}
        if not isinstance(parsed, dict):
            logger.warning("TemplateNode plan returned non-dict YAML; defaulting to empty components")
            return {"components": {}}
        return parsed

    @staticmethod
    def _sanitize_yaml_response(response: Any) -> str:
        """Strip markdown code fences or extraneous text from LLM output."""
        if response is None:
            return ""
        text = str(response).strip()
        if "```" not in text:
            return text
        match = re.search(r"```(?:yaml)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.replace("```yaml", "").replace("```", "").strip()

    async def __call__(self, state: WisbiState) -> WisbiState:
        """Execute template node: plan query and find matching templates
        
        Args:
            state: Workflow state containing query_text, llm_config, and embedder_config
            
        Returns:
            Updated state with template matching results
        """
        query_text = state.get("query_text")
        if query_text is None:
            return state
        
        # Extract components from query using LLM
        components_raw = await self._plan(state, query_text)
        components = components_raw.get("components", {}) or {}
        
        # Get embedder_config from state
        embedder_config = state.get("embedder_config")
        if embedder_config is None:
            raise ValueError("embedder_config is required in state for TemplateNode")
        
        # Find top 3 templates using RAG
        top_templates = await self._find_top_templates(
            components,
            embedder_config=embedder_config,
            top_k=3,
        )
        
        if not top_templates:
            logger.warning("No templates found matching components")
            return state
        
        # Log all template scores
        logger.info(f"Template matching results (similarity scores):")
        for idx, (score, template) in enumerate(top_templates, 1):
            logger.info(f"  Template {idx}: {template.id} - similarity: {score:.4f} (distance: {1-score:.4f})")
        
        # Store top templates in state
        state["top_templates"] = top_templates
        # Store the best match
        best_score, best_template = top_templates[0]
        state["template_score"] = best_score
        state["template"] = best_template
        state["template_sql"] = best_template.sql
        
        logger.info(f"Selected best template: {best_template.id} with confidence score: {best_score:.4f}")
        
        return state
    
    async def close(self):
        """Close the template store connection"""
        if self.template_store:
            await self.template_store.close()
