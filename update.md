# 添加 character_name 参数修改记录

## 概述
为 `generate_video` 功能添加了 `character_name` 参数，用于缓存 PTI 和 3DMM 结果，提高重复生成时的性能。

## 修改文件列表

### 1. `utils/run_nerffacespeech.py`
**修改内容：**
- 在 `generate_video` 函数签名中添加了 `character_name: str = None` 参数
- 在构建命令行参数时，如果提供了 `character_name`，会添加 `--character_name={character_name}` 参数

**具体修改：**
```python
# 修改前
def generate_video(
    audio_path: str,
    character: str,
    output_path: str,
    model_name: str
) -> bool:

# 修改后
def generate_video(
    audio_path: str,
    character: str,
    output_path: str,
    model_name: str,
    character_name: str = None
) -> bool:
```

```python
# 在 cmd 列表构建后添加
if character_name is not None:
    cmd.append(f"--character_name={character_name}")
```

---

### 2. `main.py`
**修改内容：**
- 在 `/generate_video` API 接口中添加了 `character_name` 参数（可选）
- 将 `character_name` 参数传递给 `generate_video` 函数

**具体修改：**
```python
# 修改前
@app.post("/generate_video")
def generate_video_api(
    text: str = Body(..., embed=True),
    character: str = Body(..., embed=True),
    model_name: str = Body(..., embed=True),
):

# 修改后
@app.post("/generate_video")
def generate_video_api(
    text: str = Body(..., embed=True),
    character: str = Body(..., embed=True),
    model_name: str = Body(..., embed=True),
    character_name: Optional[str] = Body(None, embed=True),
):
```

```python
# 修改前
ok2 = generate_video(
    audio_path=audio_output,
    character=character,
    output_path=video_dir,
    model_name=model_name
)

# 修改后
ok2 = generate_video(
    audio_path=audio_output,
    character=character,
    output_path=video_dir,
    model_name=model_name,
    character_name=character_name
)
```

---


