"""RAG Training Store - manages training data storage"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Union
import logging
import json

from ..connections.postgresql import PostgreSQLConfig
from ...adapters.sql.postgresql import PostgreSQLAdapter
from .schema import get_training_ddl


class TrainingStore:
    """Manages RAG training data in PostgreSQL
    
    這個類別負責管理 RAG (Retrieval-Augmented Generation) 訓練資料的儲存，
    包括問答對 (QnA)、SQL 範例 (sql_examples) 和文件說明 (documentation)。
    
    權限模型：
        - table_id: 必填，指定資料表 ID（字串）
        - user_id: 可選，空字串 = 不限使用者
        - group_id: 可選，空字串 = 不限群組
        
    存取權限規則：
        1. user_id + group_id 都有值 → 只有該使用者在該群組下可存取
        2. user_id 有值, group_id 空 → 只有該使用者可存取（跨群組）
        3. user_id 空, group_id 有值 → 該群組所有成員可存取
        4. user_id + group_id 都空 → 所有人都可存取（全局資料）
    
    使用方式：
        # 初始化實例（自動檢查並建立表）- 應用啟動時執行一次
        store = await TrainingStore.initialize(
            postgres_config=PostgreSQLConfig(
                host="localhost",
                port=5432,
                database_name="your_db",
                username="user",
                password="pass",
            ),
            training_schema="wisbi",
            embedding_dim=768,
        )
        
        # 之後所有請求都重用這個 store 實例，不需要關閉
    """
    
    def __init__(
        self,
        postgres_config: PostgreSQLConfig,
        training_schema: str = "wisbi",
        embedding_dim: int = 768,
        embedder_config: Optional[Any] = None,
    ):
        """初始化 TrainingStore

        注意：建議使用 TrainingStore.initialize() 來建立實例，
        它會自動檢查並建立所需的表。

        Args:
            postgres_config: PostgreSQL 連線配置
            training_schema: RAG training 表要存放的 schema 名稱（預設 "wisbi"）
            embedding_dim: Embedding 向量維度（預設 768）
            embedder_config: ModelConfig instance for LiteLLM embedding（可選），用於自動生成 embedding
                            Create via create_llm_config(model_name, apikey, provider)
        """
        self.postgres_config = postgres_config
        self.training_schema = training_schema
        self.embedding_dim = embedding_dim
        self.embedder_config = embedder_config
        self.logger = logging.getLogger(__name__)
        self._adapter: Optional[PostgreSQLAdapter] = None

        # Import embedding utility if config provided
        if embedder_config is not None:
            from ..utils.models import aembed_text
            self._aembed_text = aembed_text
    
    @classmethod
    async def initialize(
        cls,
        postgres_config: PostgreSQLConfig,
        training_schema: str = "wisbi",
        embedding_dim: int = 768,
        embedder_config: Optional[Any] = None,
        auto_init_tables: bool = True,
    ) -> "TrainingStore":
        """初始化 TrainingStore 實例並自動設定表

        這是推薦的初始化方式，會自動檢查並建立所需的表（如果不存在）。

        Args:
            postgres_config: PostgreSQL 連線配置
            training_schema: RAG training 表要存放的 schema 名稱（預設 "wisbi"）
            embedding_dim: Embedding 向量維度（預設 768）
            embedder_config: ModelConfig instance for LiteLLM embedding（可選），用於自動生成 embedding
                            Create via create_llm_config(model_name, apikey, provider)
            auto_init_tables: 是否自動檢查並建立表（預設 True）

        Returns:
            TrainingStore: 已初始化的實例

        Example:
            >>> from text2query.core.utils.model_configs import create_llm_config
            >>> embedder_config = create_llm_config(
            ...     model_name="text-embedding-3-small",
            ...     apikey="your-api-key",
            ...     provider="openai"
            ... )
            >>> store = await TrainingStore.initialize(
            ...     postgres_config=PostgreSQLConfig(
            ...         host="localhost",
            ...         port=5432,
            ...         database_name="your_db",
            ...         username="user",
            ...         password="pass",
            ...     ),
            ...     training_schema="wisbi",
            ...     embedding_dim=1536,  # Match your embedding model dimension
            ...     embedder_config=embedder_config,
            ... )
        """
        store = cls(postgres_config, training_schema, embedding_dim, embedder_config)
        
        if auto_init_tables:
            # 檢查表是否存在
            tables_exist = await store.check_tables_exist()
            
            # 如果有任何表不存在，就執行建立
            if not all(tables_exist.values()):
                missing_tables = [name for name, exists in tables_exist.items() if not exists]
                store.logger.info(f"Missing tables in schema '{training_schema}': {missing_tables}")
                store.logger.info("Creating training tables...")
                
                success = await store.init_training_tables()
                if success:
                    store.logger.info(f"Training tables initialized successfully in schema '{training_schema}'")
                else:
                    store.logger.warning("Failed to initialize some training tables")
            else:
                store.logger.info(f"All training tables already exist in schema '{training_schema}'")
        
        return store
    
    def _get_adapter(self) -> PostgreSQLAdapter:
        """獲取或創建 PostgreSQL adapter"""
        if self._adapter is None:
            self._adapter = PostgreSQLAdapter(self.postgres_config)
        return self._adapter
    
    # ============================================================================
    # 初始化方法
    # ============================================================================
    
    async def init_training_tables(self) -> bool:
        """初始化 RAG training 所需的表和索引
        
        這個方法會執行以下操作：
        1. 創建 schema（如果不存在）
        2. 啟用 pgvector 擴展（如果不存在）
        3. 創建 qna、sql_examples、documentation 表（如果不存在）
        4. 創建向量搜尋索引和複合索引
        
        Returns:
            bool: 成功返回 True，失敗返回 False
        """
        try:
            adapter = self._get_adapter()
            ddl_statements = get_training_ddl(
                schema_name=self.training_schema,
                embedding_dim=self.embedding_dim
            )
            
            for idx, ddl in enumerate(ddl_statements, 1):
                self.logger.debug(f"Executing DDL statement {idx}/{len(ddl_statements)}")
                result = await adapter.sql_execution(
                    ddl,
                    safe=False,  # DDL 語句需要關閉安全檢查
                    limit=None
                )
                
                if not result.get("success"):
                    error_msg = result.get("error", "Unknown error")
                    self.logger.error(f"Failed to execute DDL statement {idx}: {error_msg}")
                    # 顯示失敗的 DDL（截斷顯示）
                    ddl_preview = ddl.strip()[:200].replace('\n', ' ')
                    self.logger.error(f"Failed DDL: {ddl_preview}...")
                    return False
            
            self.logger.info(f"Successfully set up training tables in schema '{self.training_schema}'")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error setting up training tables: {e}")
            return False

    # IMPORTANT: THIS IS TEMPORARY, LATER CHANGE TO A RE-EMBED OLD CONTENT
    async def _get_existing_embedding_dim(self) -> Optional[int]:
        """Detect current embedding vector dimension from existing tables.

        Returns None if tables/column not found or cannot be parsed.
        """
        try:
            adapter = self._get_adapter()
            # Use pg_catalog to read the type modifier for the embedding column
            query = f"""
                SELECT format_type(a.atttypid, a.atttypmod) AS type_repr
                FROM pg_attribute a
                WHERE a.attrelid = '{self.training_schema}.qna'::regclass
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
            """
            result = await adapter.sql_execution(query, safe=False, limit=None)
            if not result.get("success"):
                return None
            rows = result.get("result") or []
            if not rows:
                return None
            type_repr = rows[0][0] or ""
            # Expect 'vector(XXXX)'
            import re
            m = re.search(r"vector\((\d+)\)", str(type_repr))
            if m:
                return int(m.group(1))
            return None
        except Exception:
            return None

    # IMPORTANT: THIS IS TEMPORARY, LATER CHANGE TO A RE-EMBED OLD CONTENT
    async def drop_training_tables(self) -> bool:
        """Drop training tables if they exist (non-fatal if they don't)."""
        try:
            adapter = self._get_adapter()
            drop_sql = f"""
                DROP TABLE IF EXISTS {self.training_schema}.qna CASCADE;
                DROP TABLE IF EXISTS {self.training_schema}.sql_examples CASCADE;
                DROP TABLE IF EXISTS {self.training_schema}.documentation CASCADE;
            """
            result = await adapter.sql_execution(drop_sql, safe=False, limit=None)
            return bool(result.get("success"))
        except Exception as e:
            self.logger.exception(f"Error dropping training tables: {e}")
            return False

    # IMPORTANT: THIS IS TEMPORARY, LATER CHANGE TO A RE-EMBED OLD CONTENT
    async def reinit_if_dim_mismatch(self) -> bool:
        """Ensure table embedding dimension matches configured embedding_dim.

        If a mismatch is detected, drops existing tables and re-creates them
        with the configured embedding_dim.
        """
        try:
            tables_exist = await self.check_tables_exist()
            if not any(tables_exist.values()):
                return await self.init_training_tables()

            existing_dim = await self._get_existing_embedding_dim()
            if existing_dim is None or existing_dim == self.embedding_dim:
                return True

            self.logger.warning(
                "Embedding dimension mismatch detected (existing=%s, desired=%s). Reinitializing tables...",
                existing_dim,
                self.embedding_dim,
            )
            dropped = await self.drop_training_tables()
            if not dropped:
                self.logger.error("Failed to drop existing training tables for reinit")
                return False
            return await self.init_training_tables()
        except Exception as e:
            self.logger.exception(f"Error checking/reinitializing embedding dim: {e}")
            return False
    
    async def check_tables_exist(self) -> Dict[str, bool]:
        """檢查 RAG training 表是否存在
        
        Returns:
            dict: {"qna": bool, "sql_examples": bool, "documentation": bool}
            
        Example:
            >>> tables_exist = await store.check_tables_exist()
            >>> if all(tables_exist.values()):
            ...     print("All tables exist")
            >>> else:
            ...     print(f"Missing tables: {[k for k, v in tables_exist.items() if not v]}")
        """
        try:
            adapter = self._get_adapter()
            schema = self.training_schema
            
            # 查詢指定 schema 中的表
            check_query = f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{schema}' 
                AND table_name IN ('qna', 'sql_examples', 'documentation')
            """
            
            result = await adapter.sql_execution(check_query, safe=False, limit=None)
            
            if not result.get("success"):
                self.logger.warning(f"Failed to check tables: {result.get('error', 'Unknown error')}")
                return {"qna": False, "sql_examples": False, "documentation": False}
            
            # 提取已存在的表名
            existing_tables = [row[0] for row in result.get("result", [])]
            
            return {
                "qna": "qna" in existing_tables,
                "sql_examples": "sql_examples" in existing_tables,
                "documentation": "documentation" in existing_tables,
            }
            
        except Exception as e:
            self.logger.exception(f"Error checking tables: {e}")
            return {"qna": False, "sql_examples": False, "documentation": False}
    
    async def check_schema_exists(self) -> bool:
        """檢查 training schema 是否存在
        
        Returns:
            bool: schema 存在返回 True，否則返回 False
        """
        try:
            adapter = self._get_adapter()
            check_query = f"""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name = '{self.training_schema}'
            """
            
            result = await adapter.sql_execution(check_query, safe=False, limit=None)
            
            if not result.get("success"):
                return False
            
            return len(result.get("result", [])) > 0
            
        except Exception as e:
            self.logger.exception(f"Error checking schema: {e}")
            return False
    
    # ============================================================================
    # Embedding 生成輔助方法
    # ============================================================================
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """生成文本的 embedding 向量

        Args:
            text: 要生成 embedding 的文本

        Returns:
            List[float]: embedding 向量

        Raises:
            RuntimeError: 如果沒有提供 embedder_config
        """
        if self.embedder_config is None:
            raise RuntimeError(
                "embedder_config not provided. Please initialize TrainingStore with an embedder_config "
                "or use the manual insert methods (insert_qna, insert_sql_example, insert_documentation) "
                "with pre-computed embeddings."
            )

        try:
            # Use LiteLLM aembed_text
            embedding = await self._aembed_text(self.embedder_config, text)
            return embedding

        except Exception as e:
            self.logger.exception(f"Error generating embedding: {e}")
            raise
    
    # ============================================================================
    # INSERT 方法 - 新增訓練資料（統一介面）
    # ============================================================================
    
    async def insert_training_item(
        self,
        *,
        type: str,
        table_id: str,
        user_id: str = "",
        group_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
        # QnA 特有
        question: Optional[str] = None,
        answer_sql: Optional[str] = None,
        # SQL Example/Documentation 特有
        content: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Optional[int]:
        """統一的訓練資料插入方法，自動生成 embedding

        這個方法簡化了訓練資料的插入流程，使用者只需指定 type 和相關欄位，
        系統會自動生成 embedding 並調用對應的底層方法。

        注意：使用此方法需要在初始化 TrainingStore 時提供 embedder_config。
        
        Args:
            type: 資料類型，必須是 "qna", "sql_example", "documentation" 之一
            table_id: 資料表 ID（必填，字串）
            user_id: 使用者 ID（可選，預設空字串 = 不限使用者）
            group_id: 群組 ID（可選，預設空字串 = 不限群組）
            metadata: 額外的 metadata（JSON 格式）
            is_active: 是否啟用（預設 True）
            question: 問題文字（type="qna" 時必填）
            answer_sql: 答案 SQL（type="qna" 時必填）
            content: 內容（type="sql_example" 或 "documentation" 時必填）
            title: 標題（type="documentation" 時可選）
        
        Returns:
            Optional[int]: 成功返回插入的 id，失敗返回 None
        
        Raises:
            ValueError: 如果 type 不合法或缺少必要欄位
            RuntimeError: 如果沒有提供 embedder_config
        
        Example:
            >>> # 插入問答對
            >>> await store.insert_training_item(
            ...     type="qna",
            ...     table_id="b2c5bce1-b684-4700-b3be-a9db93d71a2a",
            ...     question="查詢所有員工",
            ...     answer_sql='SELECT * FROM "b2c5bce1-b684-4700-b3be-a9db93d71a2a"',
            ...     user_id="user_123",
            ...     group_id="group_A",
            ... )
            
            >>> # 插入 SQL 範例
            >>> await store.insert_training_item(
            ...     type="sql_example",
            ...     table_id="b2c5bce1-b684-4700-b3be-a9db93d71a2a",
            ...     content='SELECT COUNT(*) FROM "b2c5bce1-b684-4700-b3be-a9db93d71a2a" WHERE hire_date IS NOT NULL',
            ... )
            
            >>> # 插入文件說明
            >>> await store.insert_training_item(
            ...     type="documentation",
            ...     table_id="b2c5bce1-b684-4700-b3be-a9db93d71a2a",
            ...     title="員工表說明",
            ...     content="員工基本資料表，包含姓名、部門等資訊",
            ... )
        """
        # 驗證 type
        valid_types = {"qna", "sql_example", "documentation"}
        if type not in valid_types:
            raise ValueError(f"Invalid type '{type}'. Must be one of: {valid_types}")
        
        # 根據 type 組合文本用於生成 embedding
        if type == "qna":
            if not question or not answer_sql:
                raise ValueError("For type='qna', both 'question' and 'answer_sql' are required")
            text_for_embedding = question
        elif type == "sql_example":
            if not content:
                raise ValueError("For type='sql_example', 'content' is required")
            text_for_embedding = content
        else:  # documentation
            if not content:
                raise ValueError("For type='documentation', 'content' is required")
            title_part = f"{title} " if title else ""
            text_for_embedding = f"{title_part}{content}"
        
        # 生成 embedding
        try:
            embedding = await self._generate_embedding(text_for_embedding)
        except Exception as e:
            self.logger.error(f"Failed to generate embedding for type={type}: {e}")
            return None
        
        # 根據 type 調用對應的底層插入方法
        if type == "qna":
            return await self.insert_qna(
                table_id=table_id,
                question=question,
                answer_sql=answer_sql,
                embedding=embedding,
                user_id=user_id,
                group_id=group_id,
                metadata=metadata,
                is_active=is_active,
            )
        elif type == "sql_example":
            return await self.insert_sql_example(
                table_id=table_id,
                content=content,
                embedding=embedding,
                user_id=user_id,
                group_id=group_id,
                metadata=metadata,
                is_active=is_active,
            )
        else:  # documentation
            return await self.insert_documentation(
                table_id=table_id,
                content=content,
                embedding=embedding,
                title=title,
                user_id=user_id,
                group_id=group_id,
                metadata=metadata,
                is_active=is_active,
            )
    
    # ============================================================================
    # INSERT 方法 - 新增訓練資料（底層方法）
    # ============================================================================
    
    async def insert_qna(
        self,
        *,
        table_id: str,
        question: str,
        answer_sql: str,
        embedding: List[float],
        user_id: str = "",
        group_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
    ) -> Optional[int]:
        """插入問答對訓練資料
        
        Args:
            table_id: 資料表 ID（必填，字串）
            question: 問題文字
            answer_sql: 答案 SQL
            embedding: 向量 embedding（長度需符合 embedding_dim）
            user_id: 使用者 ID（可選，預設空字串 = 不限使用者）
            group_id: 群組 ID（可選，預設空字串 = 不限群組）
            metadata: 額外的 metadata（JSON 格式）
            is_active: 是否啟用（預設 True）
        
        Returns:
            Optional[int]: 成功返回插入的 id，失敗返回 None
        
        權限說明：
            - user_id="user_123", group_id="group_A" → 只有 user_123 在 group_A 可存取
            - user_id="user_123", group_id="" → 只有 user_123 可存取（跨所有群組）
            - user_id="", group_id="group_A" → group_A 的所有成員可存取
            - user_id="", group_id="" → 所有人可存取（全局公開資料）
        
        Example:
            >>> # 插入個人私有資料
            >>> id = await store.insert_qna(
            ...     table_id="b2c5bce1-b684-4700-b3be-a9db93d71a2a",
            ...     question="查詢所有員工",
            ...     answer_sql='SELECT * FROM "b2c5bce1-b684-4700-b3be-a9db93d71a2a"',
            ...     embedding=[0.1, 0.2, ...],  # 768 維向量
            ...     user_id="user_123",
            ...     group_id="group_A",
            ... )
        """
        try:
            adapter = self._get_adapter()
            
            # 準備 metadata（確保是 JSON 格式）
            md = metadata or {}
            metadata_json = json.dumps(md, ensure_ascii=False)

            # 將 embedding 轉換為字串格式（pgvector 文字表示）
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'

            # 使用參數化查詢避免字串拼接錯誤
            insert_sql = f"""
                INSERT INTO {self.training_schema}.qna (
                    user_id, group_id, table_id,
                    question, answer_sql, embedding, metadata, is_active
                ) VALUES (
                    $1, $2, $3,
                    $4, $5, $6::vector, $7::jsonb, $8
                )
                RETURNING id
            """

            params = (
                user_id,
                group_id,
                table_id,
                question,
                answer_sql,
                embedding_str,
                metadata_json,
                is_active,
            )

            result = await adapter.sql_execution(insert_sql, params=params, safe=False, limit=None)
            
            if result.get("success") and result.get("result"):
                inserted_id = result["result"][0][0]
                self.logger.info(f"Inserted QnA: id={inserted_id}, table_id={table_id}")
                return int(inserted_id)
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to insert QnA: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.exception(f"Error inserting QnA: {e}")
            return None
    
    async def insert_sql_example(
        self,
        *,
        table_id: str,
        content: str,
        embedding: List[float],
        user_id: str = "",
        group_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
    ) -> Optional[int]:
        """插入 SQL 範例訓練資料
        
        Args:
            table_id: 資料表 ID（必填，字串）
            content: SQL 範例內容
            embedding: 向量 embedding
            user_id: 使用者 ID（可選）
            group_id: 群組 ID（可選）
            metadata: 額外的 metadata
            is_active: 是否啟用
        
        Returns:
            Optional[int]: 成功返回插入的 id，失敗返回 None
        """
        try:
            adapter = self._get_adapter()
            
            md = metadata or {}
            metadata_json = json.dumps(md, ensure_ascii=False)
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'

            insert_sql = f"""
                INSERT INTO {self.training_schema}.sql_examples (
                    user_id, group_id, table_id,
                    content, embedding, metadata, is_active
                ) VALUES (
                    $1, $2, $3,
                    $4, $5::vector, $6::jsonb, $7
                )
                RETURNING id
            """

            params = (
                user_id,
                group_id,
                table_id,
                content,
                embedding_str,
                metadata_json,
                is_active,
            )

            result = await adapter.sql_execution(insert_sql, params=params, safe=False, limit=None)
            
            if result.get("success") and result.get("result"):
                inserted_id = result["result"][0][0]
                self.logger.info(f"Inserted SQL example: id={inserted_id}, table_id={table_id}")
                return int(inserted_id)
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to insert SQL example: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.exception(f"Error inserting SQL example: {e}")
            return None
    
    async def insert_documentation(
        self,
        *,
        table_id: str,
        content: str,
        embedding: List[float],
        title: Optional[str] = None,
        user_id: str = "",
        group_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
    ) -> Optional[int]:
        """插入文件說明訓練資料
        
        Args:
            table_id: 資料表 ID（必填，字串）
            content: 文件內容
            embedding: 向量 embedding
            title: 文件標題（可選）
            user_id: 使用者 ID（可選）
            group_id: 群組 ID（可選）
            metadata: 額外的 metadata
            is_active: 是否啟用
        
        Returns:
            Optional[int]: 成功返回插入的 id，失敗返回 None
        """
        try:
            adapter = self._get_adapter()
            
            md = metadata or {}
            metadata_json = json.dumps(md, ensure_ascii=False)
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'

            insert_sql = f"""
                INSERT INTO {self.training_schema}.documentation (
                    user_id, group_id, table_id,
                    title, content, embedding, metadata, is_active
                ) VALUES (
                    $1, $2, $3,
                    $4, $5, $6::vector, $7::jsonb, $8
                )
                RETURNING id
            """

            params = (
                user_id,
                group_id,
                table_id,
                title,
                content,
                embedding_str,
                metadata_json,
                is_active,
            )

            result = await adapter.sql_execution(insert_sql, params=params, safe=False, limit=None)
            
            if result.get("success") and result.get("result"):
                inserted_id = result["result"][0][0]
                self.logger.info(f"Inserted documentation: id={inserted_id}, table_id={table_id}")
                return int(inserted_id)
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to insert documentation: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.exception(f"Error inserting documentation: {e}")
            return None
    
    # ============================================================================
    # SEARCH 方法 - 向量相似度搜尋
    # ============================================================================
    
    def _build_table_id_condition(self, table_id: Union[str, List[str]]) -> str:
        """構建 table_id 的 SQL 條件（支援單一或列表）"""
        if isinstance(table_id, list):
            ids = "', '".join(table_id)
            return f"table_id IN ('{ids}')"
        else:
            return f"table_id = '{table_id}'"
    
    async def search_qna(
        self,
        query_embedding: List[float],
        *,
        table_id: Union[str, List[str]],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        top_k: int = 5,
        only_active: bool = True,
    ) -> List[Dict[str, Any]]:
        """搜尋最相似的問答對訓練資料
        
        根據向量相似度搜尋，並根據權限過濾結果。
        
        Args:
            query_embedding: 查詢的向量 embedding
            table_id: 資料表 ID（單一字串或字串列表）
            user_id: 查詢者的使用者 ID（可選）
            group_id: 查詢者的群組 ID（可選）
            top_k: 返回前 K 筆最相似的結果
            only_active: 是否只返回啟用的資料
        
        Returns:
            List[Dict]: 搜尋結果列表，每筆包含所有欄位 + distance
        
        權限過濾邏輯：
            1. 如果提供 user_id + group_id → 可存取：
               - 該 user_id + group_id 的私有資料
               - 該 user_id 的跨群組資料 (group_id="")
               - 該 group_id 的群組共享資料 (user_id="")
               - 全局公開資料 (user_id="" + group_id="")
            
            2. 如果只提供 user_id → 可存取：
               - 該 user_id 的所有資料（不論 group_id）
               - 全局公開資料
            
            3. 如果只提供 group_id → 可存取：
               - 該 group_id 的群組共享資料
               - 全局公開資料
            
            4. 如果都不提供 → 只能存取：
               - 全局公開資料 (user_id="" + group_id="")
        
        Example:
            >>> # 搜尋單一表
            >>> results = await store.search_qna(
            ...     query_embedding=embedding,
            ...     table_id="b2c5bce1-b684-4700-b3be-a9db93d71a2a"
            ... )
            
            >>> # 搜尋多個指定表
            >>> results = await store.search_qna(
            ...     query_embedding=embedding,
            ...     table_id=["b2c5bce1-b684-4700-b3be-a9db93d71a2a", "f8a97f74-e446-494a-8417-b4fdefbd51c5"]
            ... )
        """
        try:
            adapter = self._get_adapter()
            
            # 將 embedding 轉換為字串格式
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # 構建 table_id 條件
            table_id_condition = self._build_table_id_condition(table_id)
            
            # 構建權限過濾條件
            # 根據提供的 user_id 和 group_id 決定可存取的資料範圍
            if user_id and group_id:
                # 情境 1：有 user_id + group_id
                # 可存取：完全匹配、user+空group、空user+group、全空
                permission_condition = f"""
                    (
                        (user_id = '{user_id}' AND group_id = '{group_id}') OR
                        (user_id = '{user_id}' AND group_id = '') OR
                        (user_id = '' AND group_id = '{group_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            elif user_id:
                # 情境 2：只有 user_id
                # 可存取：該 user 的所有資料 + 全局資料
                permission_condition = f"""
                    (
                        (user_id = '{user_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            elif group_id:
                # 情境 3：只有 group_id
                # 可存取：該 group 的共享資料 + 全局資料
                permission_condition = f"""
                    (
                        (user_id = '' AND group_id = '{group_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            else:
                # 情境 4：都沒有
                # 只能存取全局公開資料
                permission_condition = "(user_id = '' AND group_id = '')"
            
            # 構建完整的 SELECT 查詢
            select_sql = f"""
                SELECT 
                    id, user_id, group_id, table_id,
                    question, answer_sql, metadata, is_active,
                    created_at, updated_at,
                    (embedding <=> '{embedding_str}'::vector) AS distance
                FROM {self.training_schema}.qna
                WHERE {table_id_condition}
                  AND {permission_condition}
                  AND (NOT {only_active} OR is_active = TRUE)
                ORDER BY distance ASC
                LIMIT {top_k}
            """
            
            result = await adapter.sql_execution(select_sql, safe=False, limit=None)
            
            if result.get("success"):
                # 將結果轉換為字典列表
                columns = result.get("columns", [])
                rows = result.get("result", [])
                
                results = []
                for row in rows:
                    results.append(dict(zip(columns, row)))
                
                self.logger.info(f"Found {len(results)} QnA results for table_id={table_id}")
                return results
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to search QnA: {error_msg}")
                return []
                
        except Exception as e:
            self.logger.exception(f"Error searching QnA: {e}")
            return []
    
    async def search_sql_examples(
        self,
        query_embedding: List[float],
        *,
        table_id: Union[str, List[str]],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        top_k: int = 5,
        only_active: bool = True,
    ) -> List[Dict[str, Any]]:
        """搜尋最相似的 SQL 範例訓練資料
        
        Args:
            query_embedding: 查詢的向量 embedding
            table_id: 資料表 ID（單一字串或字串列表）
            user_id: 查詢者的使用者 ID（可選）
            group_id: 查詢者的群組 ID（可選）
            top_k: 返回前 K 筆最相似的結果
            only_active: 是否只返回啟用的資料
        
        權限過濾邏輯與 search_qna 相同。
        """
        try:
            adapter = self._get_adapter()
            
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # 構建 table_id 條件
            table_id_condition = self._build_table_id_condition(table_id)
            
            # 構建權限過濾條件（與 search_qna 相同邏輯）
            if user_id and group_id:
                permission_condition = f"""
                    (
                        (user_id = '{user_id}' AND group_id = '{group_id}') OR
                        (user_id = '{user_id}' AND group_id = '') OR
                        (user_id = '' AND group_id = '{group_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            elif user_id:
                permission_condition = f"""
                    (
                        (user_id = '{user_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            elif group_id:
                permission_condition = f"""
                    (
                        (user_id = '' AND group_id = '{group_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            else:
                permission_condition = "(user_id = '' AND group_id = '')"
            
            select_sql = f"""
                SELECT 
                    id, user_id, group_id, table_id,
                    content, metadata, is_active,
                    created_at, updated_at,
                    (embedding <=> '{embedding_str}'::vector) AS distance
                FROM {self.training_schema}.sql_examples
                WHERE {table_id_condition}
                  AND {permission_condition}
                  AND (NOT {only_active} OR is_active = TRUE)
                ORDER BY distance ASC
                LIMIT {top_k}
            """
            
            result = await adapter.sql_execution(select_sql, safe=False, limit=None)
            
            if result.get("success"):
                columns = result.get("columns", [])
                rows = result.get("result", [])
                results = [dict(zip(columns, row)) for row in rows]
                self.logger.info(f"Found {len(results)} SQL example results for table_id={table_id}")
                return results
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to search SQL examples: {error_msg}")
                return []
                
        except Exception as e:
            self.logger.exception(f"Error searching SQL examples: {e}")
            return []
    
    async def search_documentation(
        self,
        query_embedding: List[float],
        *,
        table_id: Union[str, List[str]],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        top_k: int = 5,
        only_active: bool = True,
    ) -> List[Dict[str, Any]]:
        """搜尋最相似的文件說明訓練資料
        
        Args:
            query_embedding: 查詢的向量 embedding
            table_id: 資料表 ID（單一字串或字串列表）
            user_id: 查詢者的使用者 ID（可選）
            group_id: 查詢者的群組 ID（可選）
            top_k: 返回前 K 筆最相似的結果
            only_active: 是否只返回啟用的資料
        
        權限過濾邏輯與 search_qna 相同。
        """
        try:
            adapter = self._get_adapter()
            
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # 構建 table_id 條件
            table_id_condition = self._build_table_id_condition(table_id)
            
            # 構建權限過濾條件（與 search_qna 相同邏輯）
            if user_id and group_id:
                permission_condition = f"""
                    (
                        (user_id = '{user_id}' AND group_id = '{group_id}') OR
                        (user_id = '{user_id}' AND group_id = '') OR
                        (user_id = '' AND group_id = '{group_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            elif user_id:
                permission_condition = f"""
                    (
                        (user_id = '{user_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            elif group_id:
                permission_condition = f"""
                    (
                        (user_id = '' AND group_id = '{group_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            else:
                permission_condition = "(user_id = '' AND group_id = '')"
            
            select_sql = f"""
                SELECT 
                    id, user_id, group_id, table_id,
                    title, content, metadata, is_active,
                    created_at, updated_at,
                    (embedding <=> '{embedding_str}'::vector) AS distance
                FROM {self.training_schema}.documentation
                WHERE {table_id_condition}
                  AND {permission_condition}
                  AND (NOT {only_active} OR is_active = TRUE)
                ORDER BY distance ASC
                LIMIT {top_k}
            """
            
            result = await adapter.sql_execution(select_sql, safe=False, limit=None)
            
            if result.get("success"):
                columns = result.get("columns", [])
                rows = result.get("result", [])
                results = [dict(zip(columns, row)) for row in rows]
                self.logger.info(f"Found {len(results)} documentation results for table_id={table_id}")
                return results
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to search documentation: {error_msg}")
                return []
                
        except Exception as e:
            self.logger.exception(f"Error searching documentation: {e}")
            return []
    
    async def search_all(
        self,
        query_embedding: List[float],
        *,
        table_id: Union[str, List[str]],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        top_k: int = 8,
        per_table_k: Optional[int] = None,
        only_active: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """搜尋所有類型的訓練資料（QnA, SQL範例, 文件）
        
        Args:
            query_embedding: 查詢的向量 embedding
            table_id: 資料表 ID（單一字串或字串列表）
            user_id: 查詢者的使用者 ID（可選）
            group_id: 查詢者的群組 ID（可選）
            top_k: 總共返回的結果數量
            per_table_k: 每個表返回的結果數量（預設為 top_k // 3）
            only_active: 是否只返回啟用的資料
        
        Returns:
            Dict: {
                "qna": [...],
                "sql_examples": [...],
                "documentation": [...]
            }
        
        Example:
            >>> # 搜尋指定表的所有類型資料
            >>> results = await store.search_all(
            ...     query_embedding=embedding,
            ...     table_id=["b2c5bce1-b684-4700-b3be-a9db93d71a2a", "f8a97f74-e446-494a-8417-b4fdefbd51c5"],
            ...     top_k=10
            ... )
        """
        per_k = per_table_k or max(2, top_k // 3)
        
        # 並行搜尋所有 3 個表
        qna_results = await self.search_qna(
            query_embedding,
            table_id=table_id,
            user_id=user_id,
            group_id=group_id,
            top_k=per_k,
            only_active=only_active,
        )
        
        sql_results = await self.search_sql_examples(
            query_embedding,
            table_id=table_id,
            user_id=user_id,
            group_id=group_id,
            top_k=per_k,
            only_active=only_active,
        )
        
        doc_results = await self.search_documentation(
            query_embedding,
            table_id=table_id,
            user_id=user_id,
            group_id=group_id,
            top_k=per_k,
            only_active=only_active,
        )
        
        return {
            "qna": qna_results,
            "sql_examples": sql_results,
            "documentation": doc_results,
        }
    
    # ============================================================================
    # UPDATE / DELETE 方法 
    # ============================================================================
    async def update_item(
        self,
        *,
        type: str,
        id: int,
        # Optional ownership fields
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        # Content fields (depends on type)
        question: Optional[str] = None,
        answer_sql: Optional[str] = None,
        content: Optional[str] = None,
        title: Optional[str] = None,
        # Other fields
        metadata: Optional[Dict[str, Any]] = None,
        is_active: Optional[bool] = None,
        # Whether to regenerate embedding when text fields change
        regenerate_embedding: bool = True,
    ) -> int:
        """Update a single training item by id.

        This method supports updating QnA, SQL example, and documentation rows.
        It can optionally regenerate the embedding when text fields change.

        Args:
            type: One of "qna", "sql_example", "documentation"
            id: Primary key id of the row to update
            user_id: New user_id (optional)
            group_id: New group_id (optional)
            question: New question text (for type="qna")
            answer_sql: New answer SQL (for type="qna")
            content: New content (for type="sql_example" or "documentation")
            title: New title (for type="documentation")
            metadata: New metadata dict (will be stored as jsonb)
            is_active: New active flag
            regenerate_embedding: If True, will regenerate embedding when text fields are provided.
                For:
                    - qna: re-embeds when a new question is provided
                    - sql_example: requires content when True
                    - documentation: requires content (title optional) when True

        Returns:
            int: number of updated rows (0 or 1)
        """
        try:
            adapter = self._get_adapter()

            if type == "qna":
                tbl = "qna"
            elif type == "sql_example":
                tbl = "sql_examples"
            elif type == "documentation":
                tbl = "documentation"
            else:
                raise ValueError(f"Unsupported type: {type}")

            set_clauses: List[str] = []
            params: List[Any] = []

            # Ownership fields
            if user_id is not None:
                set_clauses.append(f"user_id = ${len(params) + 1}")
                params.append(user_id)
            if group_id is not None:
                set_clauses.append(f"group_id = ${len(params) + 1}")
                params.append(group_id)

            # Content fields
            if tbl == "qna":
                if question is not None:
                    set_clauses.append(f"question = ${len(params) + 1}")
                    params.append(question)
                if answer_sql is not None:
                    set_clauses.append(f"answer_sql = ${len(params) + 1}")
                    params.append(answer_sql)
            elif tbl == "sql_examples":
                if content is not None:
                    set_clauses.append(f"content = ${len(params) + 1}")
                    params.append(content)
            else:  # documentation
                if title is not None:
                    set_clauses.append(f"title = ${len(params) + 1}")
                    params.append(title)
                if content is not None:
                    set_clauses.append(f"content = ${len(params) + 1}")
                    params.append(content)

            # Other fields
            if metadata is not None:
                metadata_json = json.dumps(metadata, ensure_ascii=False)
                set_clauses.append(f"metadata = ${len(params) + 1}::jsonb")
                params.append(metadata_json)
            if is_active is not None:
                set_clauses.append(f"is_active = ${len(params) + 1}")
                params.append(is_active)

            # Embedding regeneration
            if regenerate_embedding:
                text_for_embedding: Optional[str] = None

                if tbl == "qna":
                    if question is not None:
                        text_for_embedding = question
                elif tbl == "sql_examples":
                    if not content:
                        raise ValueError(
                            "For type='sql_example', 'content' is required "
                            "when regenerate_embedding=True"
                        )
                    text_for_embedding = content
                elif tbl == "documentation":
                    if not content:
                        raise ValueError(
                            "For type='documentation', 'content' is required "
                            "when regenerate_embedding=True"
                        )
                    title_part = f"{title} " if title else ""
                    text_for_embedding = f"{title_part}{content}"
                else:
                    raise ValueError(f"Unsupported type: {tbl}")

                if text_for_embedding is not None:
                    embedding = await self._generate_embedding(text_for_embedding)
                    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                    set_clauses.append(f"embedding = ${len(params) + 1}::vector")
                    params.append(embedding_str)

            if not set_clauses:
                # Nothing to update
                self.logger.info(
                    f"No fields provided to update for {self.training_schema}.{tbl} id={id}"
                )
                return 0

            # Always update updated_at
            set_clauses.append("updated_at = now()")

            set_sql = ", ".join(set_clauses)
            where_param_idx = len(params) + 1
            update_sql = f"""
                UPDATE {self.training_schema}.{tbl}
                SET {set_sql}
                WHERE id = ${where_param_idx}
                RETURNING id
            """
            params.append(id)

            result = await adapter.sql_execution(update_sql, params=tuple(params), safe=False, limit=None)
            if result.get("success"):
                rows = result.get("result", []) or []
                updated = len(rows)
                self.logger.info(
                    f"Updated {updated} row(s) in {self.training_schema}.{tbl} with id={id}"
                )
                return updated
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to update item: {error_msg}")
                return 0
        except Exception as e:
            self.logger.exception(f"Error updating item: {e}")
            return 0

    async def delete_item(
        self,
        *,
        type: str,
        table_id: str,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        question: Optional[str] = None,
        answer_sql: Optional[str] = None,
        content: Optional[str] = None,
        title: Optional[str] = None,
    ) -> int:
        """Delete training item(s) by available fields only.
        
        Uses existing columns: table_id, user_id, group_id, question, answer_sql, content, title.
        Text comparisons are exact matches.
        
        Args:
            type: One of "qna", "sql_example", "documentation"
            table_id: Scope to the datasource/table
            user_id: Optional scope
            group_id: Optional scope
            question: For qna rows (optional but recommended)
            answer_sql: For qna rows (optional but recommended)
            content: For sql_example/documentation rows
            title: For documentation rows (optional)
        
        Returns:
            int: number of deleted rows
        """
        try:
            adapter = self._get_adapter()
            if type == "qna":
                tbl = "qna"
            elif type == "sql_example":
                tbl = "sql_examples"
            elif type == "documentation":
                tbl = "documentation"
            else:
                raise ValueError(f"Unsupported type: {type}")

            where_clauses: List[str] = ["table_id = '{}'".format(table_id)]
            if user_id is not None and user_id != "":
                where_clauses.append("user_id = '{}'".format(user_id))
            if group_id is not None and group_id != "":
                where_clauses.append("group_id = '{}'".format(group_id))

            if tbl == "qna":
                if question:
                    where_clauses.append("question = $${}$$".format(question))
                if answer_sql:
                    where_clauses.append("answer_sql = $${}$$".format(answer_sql))
            elif tbl == "sql_examples":
                if content:
                    where_clauses.append("content = $${}$$".format(content))
            else:  # documentation
                if content:
                    where_clauses.append("content = $${}$$".format(content))
                if title:
                    where_clauses.append("title = $${}$$".format(title))

            where_sql = " AND ".join(where_clauses)
            delete_sql = f"""
                DELETE FROM {self.training_schema}.{tbl}
                WHERE {where_sql}
                RETURNING id
            """

            result = await adapter.sql_execution(delete_sql, safe=False, limit=None)
            if result.get("success"):
                rows = result.get("result", []) or []
                deleted = len(rows)
                self.logger.info(
                    f"Deleted {deleted} row(s) from {self.training_schema}.{tbl} by fields"
                )
                return deleted
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to delete by fields: {error_msg}")
                return 0
        except Exception as e:
            self.logger.exception(f"Error deleting by fields: {e}")
            return 0
    
    # ============================================================================
    # 連線管理
    # ============================================================================
    
    async def close(self):
        """關閉資料庫連線
        
        在應用程式結束時應該呼叫這個方法來釋放資源。
        注意：在後端服務中，通常不需要手動呼叫，連線池會自動管理。
        
        Example:
            >>> store = await TrainingStore.initialize(...)
            >>> try:
            ...     # 使用 store
            ...     pass
            ... finally:
            ...     await store.close()
        """
        if self._adapter:
            await self._adapter.close_pool()
            self.logger.debug("TrainingStore connection closed")