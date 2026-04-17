"""Tool execution simulator with deterministic mock outputs."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from toolblind.utils.logging import get_logger

logger = get_logger("simulator")


class ToolUnavailableError(Exception):
    """Raised when an agent tries to call an unavailable tool."""

    def __init__(self, tool_name: str, reason: str):
        """Initialize with tool name and unavailability reason."""
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Tool '{tool_name}' is unavailable: {reason}")


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    output: Any
    output_type: str
    latency_ms: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "output_type": self.output_type,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolResult":
        """Deserialize from dictionary."""
        return cls(
            success=d["success"],
            output=d["output"],
            output_type=d["output_type"],
            latency_ms=d["latency_ms"],
        )


def _deterministic_seed(tool_name: str, params: Dict) -> int:
    """Create a deterministic integer seed from tool name and params."""
    raw = f"{tool_name}:{json.dumps(params, sort_keys=True)}"
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)


class ToolSimulator:
    """Deterministic mock backend for tool execution."""

    def __init__(self, unavailable_tools: Optional[Set[str]] = None,
                 unavailability_reasons: Optional[Dict[str, str]] = None):
        """Initialize the simulator with a set of unavailable tools."""
        self._unavailable: Set[str] = unavailable_tools or set()
        self._reasons: Dict[str, str] = unavailability_reasons or {}
        self._call_log: list = []

    def execute(self, tool_name: str, params: Dict, task_context: Optional[Dict] = None) -> ToolResult:
        """Execute a tool call against the mock backend."""
        if tool_name in self._unavailable:
            reason = self._reasons.get(tool_name, "Tool is unavailable")
            raise ToolUnavailableError(tool_name, reason)

        seed = _deterministic_seed(tool_name, params)
        result = self._generate_mock_output(tool_name, params, seed)
        self._call_log.append({
            "tool": tool_name,
            "params": params,
            "result": result.to_dict(),
        })
        return result

    def get_call_log(self) -> list:
        """Return the log of all tool calls made."""
        return list(self._call_log)

    def _generate_mock_output(self, tool_name: str, params: Dict, seed: int) -> ToolResult:
        """Generate a realistic deterministic mock output based on tool name."""
        generators = {
            # Web domain
            "fetch_url": self._mock_fetch_url,
            "search_web": self._mock_search_web,
            "parse_html": self._mock_parse_html,
            "extract_links": self._mock_extract_links,
            "extract_text": self._mock_extract_text,
            "screenshot_page": self._mock_screenshot_page,
            "check_status": self._mock_check_status,
            "download_file": self._mock_download_file,
            "submit_form": self._mock_submit_form,
            "get_headers": self._mock_get_headers,
            # Code domain
            "read_file": self._mock_read_file,
            "write_file": self._mock_write_file,
            "execute_python": self._mock_execute_python,
            "execute_bash": self._mock_execute_bash,
            "lint_code": self._mock_lint_code,
            "format_code": self._mock_format_code,
            "parse_ast": self._mock_parse_ast,
            "git_commit": self._mock_git_commit,
            "install_package": self._mock_install_package,
            "run_tests": self._mock_run_tests,
            # File domain
            "read_file_content": self._mock_read_file,
            "write_file_content": self._mock_write_file,
            "list_directory": self._mock_list_directory,
            "copy_file": self._mock_copy_file,
            "delete_file": self._mock_delete_file,
            "compress_file": self._mock_compress_file,
            "extract_archive": self._mock_extract_archive,
            "get_metadata": self._mock_get_metadata,
            "convert_format": self._mock_convert_format,
            "search_content": self._mock_search_content,
            # API domain
            "http_get": self._mock_http_get,
            "http_post": self._mock_http_post,
            "parse_json": self._mock_parse_json,
            "authenticate_oauth": self._mock_authenticate_oauth,
            "rate_limit_wait": self._mock_rate_limit_wait,
            "cache_response": self._mock_cache_response,
            "validate_schema": self._mock_validate_schema,
            "retry_request": self._mock_retry_request,
            "log_request": self._mock_log_request,
            "transform_response": self._mock_transform_response,
            # Database domain
            "sql_query": self._mock_sql_query,
            "sql_insert": self._mock_sql_insert,
            "sql_update": self._mock_sql_update,
            "connect_db": self._mock_connect_db,
            "list_tables": self._mock_list_tables,
            "get_schema": self._mock_get_schema,
            "export_csv": self._mock_export_csv,
            "run_migration": self._mock_run_migration,
            "backup_table": self._mock_backup_table,
            "check_constraints": self._mock_check_constraints,
        }

        generator = generators.get(tool_name, self._mock_generic)
        return generator(params, seed)

    # -- Web mocks --

    def _mock_fetch_url(self, params: Dict, seed: int) -> ToolResult:
        """Mock fetching a URL and returning HTML."""
        url = params.get("url", "https://example.com")
        html = (
            f'<!DOCTYPE html><html><head><title>Page at {url}</title></head>'
            f'<body><h1>Welcome</h1><p>Content from {url} loaded successfully.</p>'
            f'<a href="https://example.com/about">About</a>'
            f'<a href="https://example.com/contact">Contact</a>'
            f'<div class="main"><p>Main content section with id={seed}.</p></div>'
            f'</body></html>'
        )
        return ToolResult(success=True, output=html, output_type="html_string", latency_ms=120 + seed % 200)

    def _mock_search_web(self, params: Dict, seed: int) -> ToolResult:
        """Mock web search returning result list."""
        query = params.get("query", "search")
        num = params.get("num_results", 5)
        results = []
        for i in range(min(num, 10)):
            results.append({
                "url": f"https://result{i + 1}.example.com/{query.replace(' ', '-')}",
                "title": f"Result {i + 1} for '{query}'",
                "snippet": f"This page contains information about {query}. Relevant section {seed + i}.",
            })
        return ToolResult(success=True, output=results, output_type="search_results", latency_ms=250 + seed % 300)

    def _mock_parse_html(self, params: Dict, seed: int) -> ToolResult:
        """Mock HTML parsing returning elements."""
        selector = params.get("selector", "div")
        elements = [
            {"tag": selector, "text": f"Element content block {i}", "attrs": {"class": f"item-{i}"}}
            for i in range(3 + seed % 5)
        ]
        return ToolResult(success=True, output=elements, output_type="element_list", latency_ms=15 + seed % 30)

    def _mock_extract_links(self, params: Dict, seed: int) -> ToolResult:
        """Mock link extraction."""
        links = [
            f"https://example.com/page{i}" for i in range(5 + seed % 10)
        ]
        return ToolResult(success=True, output=links, output_type="url_list", latency_ms=10 + seed % 20)

    def _mock_extract_text(self, params: Dict, seed: int) -> ToolResult:
        """Mock text extraction from HTML."""
        text = (
            f"Welcome. Content loaded successfully. Main content section with id={seed}. "
            f"This document contains {3 + seed % 10} paragraphs of text about the topic. "
            f"The information was last updated on 2024-01-{1 + seed % 28:02d}."
        )
        return ToolResult(success=True, output=text, output_type="plain_text", latency_ms=8 + seed % 15)

    def _mock_screenshot_page(self, params: Dict, seed: int) -> ToolResult:
        """Mock page screenshot."""
        width = params.get("width", 1280)
        height = params.get("height", 720)
        output = {"format": "png", "width": width, "height": height, "size_bytes": 150000 + seed % 50000}
        return ToolResult(success=True, output=output, output_type="image_bytes", latency_ms=800 + seed % 500)

    def _mock_check_status(self, params: Dict, seed: int) -> ToolResult:
        """Mock HTTP status check."""
        codes = [200, 200, 200, 301, 404, 500]
        code = codes[seed % len(codes)]
        output = {"status_code": code, "url": params.get("url", ""), "headers": {"content-type": "text/html"}}
        return ToolResult(success=True, output=output, output_type="status_info", latency_ms=50 + seed % 100)

    def _mock_download_file(self, params: Dict, seed: int) -> ToolResult:
        """Mock file download."""
        path = params.get("save_path", f"/tmp/download_{seed}.dat")
        return ToolResult(success=True, output=path, output_type="file_path", latency_ms=500 + seed % 2000)

    def _mock_submit_form(self, params: Dict, seed: int) -> ToolResult:
        """Mock form submission."""
        output = {
            "status_code": 200,
            "body": f"Form submitted successfully. Confirmation ID: CF-{seed:08d}",
            "headers": {"content-type": "text/html"},
        }
        return ToolResult(success=True, output=output, output_type="http_response", latency_ms=200 + seed % 300)

    def _mock_get_headers(self, params: Dict, seed: int) -> ToolResult:
        """Mock getting HTTP headers."""
        headers = {
            "content-type": "text/html; charset=utf-8",
            "content-length": str(5000 + seed % 50000),
            "server": "nginx/1.24.0",
            "x-request-id": f"req-{seed:08x}",
            "cache-control": "max-age=3600",
        }
        return ToolResult(success=True, output=headers, output_type="header_dict", latency_ms=30 + seed % 60)

    # -- Code mocks --

    def _mock_read_file(self, params: Dict, seed: int) -> ToolResult:
        """Mock reading a file."""
        path = params.get("path", "unknown.py")
        content = (
            f'"""Module loaded from {path}."""\n\n'
            f"import os\nimport sys\n\n"
            f"def main():\n"
            f'    print("Running process {seed}")\n'
            f"    data = list(range({10 + seed % 100}))\n"
            f"    return sum(data)\n\n"
            f'if __name__ == "__main__":\n'
            f"    main()\n"
        )
        return ToolResult(success=True, output=content, output_type="file_content", latency_ms=5 + seed % 10)

    def _mock_write_file(self, params: Dict, seed: int) -> ToolResult:
        """Mock writing a file."""
        path = params.get("path", "output.txt")
        size = len(params.get("content", ""))
        return ToolResult(
            success=True,
            output={"path": path, "bytes_written": size, "status": "ok"},
            output_type="write_confirmation",
            latency_ms=5 + seed % 10,
        )

    def _mock_execute_python(self, params: Dict, seed: int) -> ToolResult:
        """Mock Python execution."""
        output = {
            "stdout": f"Process completed. Result: {seed * 7}\nTotal items processed: {10 + seed % 50}\n",
            "stderr": "",
            "exit_code": 0,
            "execution_time_ms": 100 + seed % 500,
        }
        return ToolResult(success=True, output=output, output_type="execution_result", latency_ms=200 + seed % 800)

    def _mock_execute_bash(self, params: Dict, seed: int) -> ToolResult:
        """Mock bash execution."""
        cmd = params.get("command", "echo hello")
        output = {
            "stdout": f"Command '{cmd}' executed. Output line {seed}.\n",
            "stderr": "",
            "exit_code": 0,
        }
        return ToolResult(success=True, output=output, output_type="execution_result", latency_ms=50 + seed % 200)

    def _mock_lint_code(self, params: Dict, seed: int) -> ToolResult:
        """Mock code linting."""
        issues = []
        for i in range(seed % 4):
            issues.append({
                "line": 5 + i * 3,
                "column": 1,
                "severity": "warning",
                "message": f"Line too long ({80 + i * 5} > 79 characters)",
                "rule": "E501",
            })
        return ToolResult(
            success=True,
            output={"issues": issues, "total_issues": len(issues), "passed": len(issues) == 0},
            output_type="lint_results",
            latency_ms=50 + seed % 100,
        )

    def _mock_format_code(self, params: Dict, seed: int) -> ToolResult:
        """Mock code formatting."""
        code = params.get("code", "def f(): pass")
        formatted = code.strip() + "\n"
        return ToolResult(success=True, output=formatted, output_type="formatted_code", latency_ms=20 + seed % 50)

    def _mock_parse_ast(self, params: Dict, seed: int) -> ToolResult:
        """Mock AST parsing."""
        ast = {
            "type": "Module",
            "body": [
                {"type": "FunctionDef", "name": "main", "args": [], "lineno": 4},
                {"type": "Import", "names": ["os", "sys"], "lineno": 1},
            ],
            "node_count": 8 + seed % 20,
        }
        return ToolResult(success=True, output=ast, output_type="ast_dict", latency_ms=15 + seed % 30)

    def _mock_git_commit(self, params: Dict, seed: int) -> ToolResult:
        """Mock git commit."""
        commit_hash = f"{seed:040x}"
        return ToolResult(success=True, output=commit_hash, output_type="commit_hash", latency_ms=100 + seed % 200)

    def _mock_install_package(self, params: Dict, seed: int) -> ToolResult:
        """Mock package installation."""
        pkg = params.get("package_name", "unknown")
        version = params.get("version", "latest")
        output = {"package": pkg, "version": version, "status": "installed", "dependencies_installed": 3 + seed % 5}
        return ToolResult(success=True, output=output, output_type="install_status", latency_ms=2000 + seed % 3000)

    def _mock_run_tests(self, params: Dict, seed: int) -> ToolResult:
        """Mock test execution."""
        total = 10 + seed % 20
        passed = total - seed % 3
        output = {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "skipped": 0,
            "duration_seconds": 1.5 + (seed % 100) / 10,
        }
        return ToolResult(success=True, output=output, output_type="test_results", latency_ms=3000 + seed % 5000)

    # -- File mocks --

    def _mock_list_directory(self, params: Dict, seed: int) -> ToolResult:
        """Mock directory listing."""
        ext = params.get("extension_filter", "")
        files = [
            {"name": f"file_{i}.{ext or 'txt'}", "size": 1024 * (i + 1), "type": "file"}
            for i in range(5 + seed % 10)
        ]
        files.append({"name": "subdir", "size": 0, "type": "directory"})
        return ToolResult(success=True, output=files, output_type="file_list", latency_ms=10 + seed % 20)

    def _mock_copy_file(self, params: Dict, seed: int) -> ToolResult:
        """Mock file copy."""
        return ToolResult(
            success=True,
            output={"source": params.get("source", ""), "destination": params.get("destination", ""), "status": "copied"},
            output_type="copy_confirmation",
            latency_ms=20 + seed % 50,
        )

    def _mock_delete_file(self, params: Dict, seed: int) -> ToolResult:
        """Mock file deletion."""
        return ToolResult(
            success=True,
            output={"path": params.get("path", ""), "status": "deleted"},
            output_type="delete_confirmation",
            latency_ms=5 + seed % 10,
        )

    def _mock_compress_file(self, params: Dict, seed: int) -> ToolResult:
        """Mock file compression."""
        archive = params.get("archive_path", f"/tmp/archive_{seed}.zip")
        return ToolResult(
            success=True,
            output=archive,
            output_type="archive_path",
            latency_ms=200 + seed % 500,
        )

    def _mock_extract_archive(self, params: Dict, seed: int) -> ToolResult:
        """Mock archive extraction."""
        dest = params.get("destination", f"/tmp/extracted_{seed}")
        return ToolResult(
            success=True,
            output=dest,
            output_type="extracted_path",
            latency_ms=300 + seed % 400,
        )

    def _mock_get_metadata(self, params: Dict, seed: int) -> ToolResult:
        """Mock file metadata retrieval."""
        metadata = {
            "path": params.get("path", ""),
            "size_bytes": 4096 * (1 + seed % 1000),
            "created": "2024-01-15T10:30:00Z",
            "modified": f"2024-0{1 + seed % 9}-{1 + seed % 28:02d}T14:22:00Z",
            "permissions": "rw-r--r--",
            "owner": "user",
        }
        return ToolResult(success=True, output=metadata, output_type="metadata_dict", latency_ms=5 + seed % 10)

    def _mock_convert_format(self, params: Dict, seed: int) -> ToolResult:
        """Mock format conversion."""
        output_path = params.get("output_path", f"/tmp/converted_{seed}")
        return ToolResult(success=True, output=output_path, output_type="converted_path", latency_ms=100 + seed % 300)

    def _mock_search_content(self, params: Dict, seed: int) -> ToolResult:
        """Mock content search."""
        pattern = params.get("pattern", ".*")
        matches = [
            {"line_number": 10 + i * 5, "content": f"Line matching '{pattern}': data_{seed + i}"}
            for i in range(2 + seed % 5)
        ]
        return ToolResult(success=True, output=matches, output_type="match_list", latency_ms=30 + seed % 100)

    # -- API mocks --

    def _mock_http_get(self, params: Dict, seed: int) -> ToolResult:
        """Mock HTTP GET request."""
        output = {
            "status_code": 200,
            "body": {"data": [{"id": i, "value": f"item_{seed + i}"} for i in range(5)], "total": 42 + seed % 100},
            "headers": {"content-type": "application/json", "x-request-id": f"req-{seed:08x}"},
        }
        return ToolResult(success=True, output=output, output_type="http_response", latency_ms=150 + seed % 300)

    def _mock_http_post(self, params: Dict, seed: int) -> ToolResult:
        """Mock HTTP POST request."""
        output = {
            "status_code": 201,
            "body": {"id": f"created-{seed:08x}", "status": "created"},
            "headers": {"content-type": "application/json", "location": f"/resources/created-{seed:08x}"},
        }
        return ToolResult(success=True, output=output, output_type="http_response", latency_ms=200 + seed % 400)

    def _mock_parse_json(self, params: Dict, seed: int) -> ToolResult:
        """Mock JSON parsing."""
        parsed = {"records": [{"key": f"k{i}", "value": seed + i} for i in range(3)], "meta": {"count": 3}}
        return ToolResult(success=True, output=parsed, output_type="parsed_data", latency_ms=2 + seed % 5)

    def _mock_authenticate_oauth(self, params: Dict, seed: int) -> ToolResult:
        """Mock OAuth authentication."""
        token = f"eyJhbGciOiJSUzI1NiJ9.{seed:032x}.signature"
        return ToolResult(success=True, output=token, output_type="access_token", latency_ms=500 + seed % 500)

    def _mock_rate_limit_wait(self, params: Dict, seed: int) -> ToolResult:
        """Mock rate limit wait."""
        wait = params.get("retry_after", 1)
        return ToolResult(
            success=True,
            output={"waited_seconds": wait, "status": "ready"},
            output_type="wait_confirmation",
            latency_ms=wait * 1000,
        )

    def _mock_cache_response(self, params: Dict, seed: int) -> ToolResult:
        """Mock response caching."""
        key = params.get("key", f"cache-{seed}")
        return ToolResult(
            success=True,
            output={"key": key, "status": "cached", "ttl_remaining": params.get("ttl_seconds", 3600)},
            output_type="cache_confirmation",
            latency_ms=5 + seed % 10,
        )

    def _mock_validate_schema(self, params: Dict, seed: int) -> ToolResult:
        """Mock schema validation."""
        return ToolResult(
            success=True,
            output={"valid": True, "errors": [], "warnings": []},
            output_type="validation_result",
            latency_ms=10 + seed % 20,
        )

    def _mock_retry_request(self, params: Dict, seed: int) -> ToolResult:
        """Mock retry request."""
        output = {
            "status_code": 200,
            "body": {"data": "retried_successfully", "attempt": 1 + seed % 3},
            "headers": {"content-type": "application/json"},
        }
        return ToolResult(success=True, output=output, output_type="http_response", latency_ms=300 + seed % 600)

    def _mock_log_request(self, params: Dict, seed: int) -> ToolResult:
        """Mock request logging."""
        return ToolResult(
            success=True,
            output={"logged": True, "log_path": params.get("log_path", "/var/log/api.log"), "entry_id": seed},
            output_type="log_confirmation",
            latency_ms=5 + seed % 10,
        )

    def _mock_transform_response(self, params: Dict, seed: int) -> ToolResult:
        """Mock response transformation."""
        return ToolResult(
            success=True,
            output={"transformed": True, "fields_extracted": 3 + seed % 5, "data": {"result": seed}},
            output_type="transformed_data",
            latency_ms=5 + seed % 15,
        )

    # -- Database mocks --

    def _mock_sql_query(self, params: Dict, seed: int) -> ToolResult:
        """Mock SQL SELECT query."""
        rows = [
            {"id": i + 1, "name": f"record_{seed + i}", "value": (seed + i) * 3.14, "active": i % 2 == 0}
            for i in range(5 + seed % 10)
        ]
        return ToolResult(
            success=True,
            output={"rows": rows, "row_count": len(rows), "columns": ["id", "name", "value", "active"]},
            output_type="query_results",
            latency_ms=20 + seed % 100,
        )

    def _mock_sql_insert(self, params: Dict, seed: int) -> ToolResult:
        """Mock SQL INSERT."""
        rows = params.get("rows", [{}])
        return ToolResult(
            success=True, output={"rows_inserted": len(rows)}, output_type="insert_count", latency_ms=15 + seed % 50
        )

    def _mock_sql_update(self, params: Dict, seed: int) -> ToolResult:
        """Mock SQL UPDATE."""
        return ToolResult(
            success=True, output={"rows_affected": 1 + seed % 10}, output_type="update_count", latency_ms=15 + seed % 50
        )

    def _mock_connect_db(self, params: Dict, seed: int) -> ToolResult:
        """Mock database connection."""
        return ToolResult(
            success=True,
            output={"handle": f"conn-{seed:08x}", "database": "main", "status": "connected"},
            output_type="connection_handle",
            latency_ms=100 + seed % 200,
        )

    def _mock_list_tables(self, params: Dict, seed: int) -> ToolResult:
        """Mock listing tables."""
        tables = [
            {"name": name, "row_count": 100 * (i + 1) + seed % 1000}
            for i, name in enumerate(["users", "orders", "products", "sessions", "audit_log"])
        ]
        return ToolResult(success=True, output=tables, output_type="table_list", latency_ms=10 + seed % 30)

    def _mock_get_schema(self, params: Dict, seed: int) -> ToolResult:
        """Mock getting table schema."""
        table = params.get("table", "unknown")
        schema = {
            "table": table,
            "columns": [
                {"name": "id", "type": "INTEGER", "nullable": False, "primary_key": True},
                {"name": "name", "type": "VARCHAR(255)", "nullable": False, "primary_key": False},
                {"name": "created_at", "type": "TIMESTAMP", "nullable": False, "primary_key": False},
                {"name": "data", "type": "JSONB", "nullable": True, "primary_key": False},
            ],
            "indexes": [f"idx_{table}_id", f"idx_{table}_name"],
        }
        return ToolResult(success=True, output=schema, output_type="schema_info", latency_ms=10 + seed % 20)

    def _mock_export_csv(self, params: Dict, seed: int) -> ToolResult:
        """Mock CSV export."""
        path = params.get("output_path", f"/tmp/export_{seed}.csv")
        return ToolResult(
            success=True,
            output={"path": path, "rows_exported": 50 + seed % 500, "size_bytes": 4096 + seed % 100000},
            output_type="csv_path",
            latency_ms=100 + seed % 300,
        )

    def _mock_run_migration(self, params: Dict, seed: int) -> ToolResult:
        """Mock database migration."""
        direction = params.get("direction", "up")
        return ToolResult(
            success=True,
            output={"direction": direction, "status": "applied", "tables_affected": 1 + seed % 3},
            output_type="migration_result",
            latency_ms=500 + seed % 1000,
        )

    def _mock_backup_table(self, params: Dict, seed: int) -> ToolResult:
        """Mock table backup."""
        table = params.get("table", "unknown")
        backup_name = params.get("backup_name", f"{table}_backup_{seed}")
        return ToolResult(
            success=True,
            output={"backup_name": backup_name, "rows_backed_up": 100 + seed % 1000, "status": "complete"},
            output_type="backup_confirmation",
            latency_ms=300 + seed % 500,
        )

    def _mock_check_constraints(self, params: Dict, seed: int) -> ToolResult:
        """Mock constraint checking."""
        return ToolResult(
            success=True,
            output={
                "table": params.get("table", "unknown"),
                "violations": [],
                "constraints_checked": 4 + seed % 3,
                "all_valid": True,
            },
            output_type="constraint_report",
            latency_ms=20 + seed % 50,
        )

    def _mock_generic(self, params: Dict, seed: int) -> ToolResult:
        """Fallback mock for any unregistered tool."""
        return ToolResult(
            success=True,
            output={"status": "completed", "result": f"generic_output_{seed}", "params_received": list(params.keys())},
            output_type="generic_result",
            latency_ms=50 + seed % 200,
        )
