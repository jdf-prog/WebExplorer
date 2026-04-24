import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union


class QwenCodeInterpreterTool:
    name = "code_interpreter"
    description = "Python code sandbox, which can be used to execute Python code."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The python code.",
            }
        },
        "required": ["code"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.timeout = int(cfg.get("timeout", os.getenv("QWEN_CODE_TIMEOUT", "30")))
        self.workdir = Path(cfg.get("workdir", os.getenv("QWEN_CODE_WORKDIR", "."))).resolve()

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            code = params["code"]
        except Exception:
            return "[code_interpreter] Invalid request format: missing 'code'"

        if not isinstance(code, str) or not code.strip():
            return "[code_interpreter] Error: 'code' must be a non-empty string"

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8",
                dir=self.workdir,
            ) as temp_file:
                temp_file.write(code)
                temp_path = Path(temp_file.name)

            result = subprocess.run(
                [sys.executable, str(temp_path)],
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            payload = {
                "returncode": result.returncode,
                "stdout": result.stdout[-12000:],
                "stderr": result.stderr[-12000:],
            }
            return json.dumps(payload, ensure_ascii=False)
        except subprocess.TimeoutExpired as exc:
            payload = {
                "error": f"Execution timed out after {self.timeout} seconds",
                "stdout": (exc.stdout or "")[-12000:],
                "stderr": (exc.stderr or "")[-12000:],
            }
            return json.dumps(payload, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def make_tool_message(self, content: str) -> dict:
        return {"role": "tool", "name": self.name, "content": content}
