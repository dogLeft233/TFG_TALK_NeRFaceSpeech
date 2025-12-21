#!/usr/bin/env python3
"""
Gradio 前端应用测试程序
测试前端应用的导入和基本功能
"""
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_imports():
    """测试模块导入"""
    print("\n" + "="*60)
    print("测试模块导入")
    print("="*60)
    
    results = []
    
    # 测试导入 gradio
    try:
        import gradio as gr
        print(f"   ✓ gradio 导入成功 (版本: {gr.__version__})")
        results.append(("gradio", True))
    except ImportError as e:
        print(f"   ✗ gradio 导入失败: {e}")
        print(f"      请安装: pip install gradio")
        results.append(("gradio", False))
    
    # 测试导入 requests
    try:
        import requests
        print(f"   ✓ requests 导入成功")
        results.append(("requests", True))
    except ImportError as e:
        print(f"   ✗ requests 导入失败: {e}")
        results.append(("requests", False))
    
    # 测试导入 frontend.app
    try:
        from frontend import app
        print(f"   ✓ frontend.app 导入成功")
        results.append(("frontend.app", True))
    except ImportError as e:
        print(f"   ✗ frontend.app 导入失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("frontend.app", False))
    
    return results


def test_dependencies():
    """测试依赖模块"""
    print("\n" + "="*60)
    print("测试依赖模块")
    print("="*60)
    
    results = []
    
    # 测试 shared.config
    try:
        from shared.config import PROJECT_ROOT, MODEL_DIR
        print(f"   ✓ shared.config 导入成功")
        print(f"      PROJECT_ROOT: {PROJECT_ROOT}")
        
        # 测试 get_character_list（如果存在）
        try:
            from shared.config import get_character_list
            characters = get_character_list()
            print(f"      角色列表: {characters}")
        except ImportError:
            print(f"      ⚠️  get_character_list 函数不存在，使用默认实现")
        
        results.append(("shared.config", True))
    except Exception as e:
        print(f"   ✗ shared.config 导入失败: {e}")
        results.append(("shared.config", False))
    
    # 测试 shared.database
    try:
        from shared.database import settings_db, video_records_db, chat_db
        print(f"   ✓ shared.database 导入成功")
        results.append(("shared.database", True))
    except Exception as e:
        print(f"   ✗ shared.database 导入失败: {e}")
        results.append(("shared.database", False))
    
    return results


def test_app_creation():
    """测试应用创建"""
    print("\n" + "="*60)
    print("测试应用创建")
    print("="*60)
    
    try:
        import gradio as gr
    except ImportError:
        print("   ⚠️  Gradio 未安装，跳过此测试")
        return None
    
    try:
        from frontend.app import create_main_app
        app = create_main_app()
        assert app is not None, "app 对象不存在"
        print(f"   ✓ Gradio 应用创建成功")
        print(f"   ✓ app 类型: {type(app)}")
        return True
    except Exception as e:
        print(f"   ✗ Gradio 应用创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_helper_functions():
    """测试辅助函数"""
    print("\n" + "="*60)
    print("测试辅助函数")
    print("="*60)
    
    results = []
    
    try:
        # 直接读取文件内容，不导入（避免 Gradio 依赖）
        app_file = Path(__file__).parent.parent / "frontend" / "app.py"
        if not app_file.exists():
            print(f"   ✗ frontend/app.py 文件不存在")
            results.append(("helper_functions", False))
            return results
        
        # 检查函数是否存在
        with open(app_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'def get_settings' in content:
            print(f"   ✓ get_settings 函数存在")
            results.append(("get_settings", True))
        else:
            print(f"   ✗ get_settings 函数不存在")
            results.append(("get_settings", False))
        
        if 'def get_models' in content:
            print(f"   ✓ get_models 函数存在")
            results.append(("get_models", True))
        else:
            print(f"   ✗ get_models 函数不存在")
            results.append(("get_models", False))
        
        # 尝试导入（如果 Gradio 可用）
        try:
            from frontend.app import get_settings, get_models
            
            # 测试 get_settings
            settings = get_settings()
            print(f"   ✓ get_settings 执行成功，返回 {len(settings)} 个设置")
            
            # 测试 get_models
            models = get_models()
            print(f"   ✓ get_models 执行成功，找到 {len(models)} 个模型")
            if models:
                print(f"      模型列表: {models[:3]}...")  # 只显示前3个
        except ImportError as e:
            if 'gradio' in str(e).lower():
                print(f"   ⚠️  Gradio 未安装，跳过函数执行测试")
            else:
                raise
        
    except Exception as e:
        print(f"   ✗ 辅助函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("helper_functions", False))
    
    return results


def test_css_styles():
    """测试 CSS 样式"""
    print("\n" + "="*60)
    print("测试 CSS 样式")
    print("="*60)
    
    try:
        # 直接读取文件内容，不导入（避免 Gradio 依赖）
        app_file = Path(__file__).parent.parent / "frontend" / "app.py"
        if not app_file.exists():
            print(f"   ✗ frontend/app.py 文件不存在")
            return False
        
        with open(app_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取 CUSTOM_CSS
        import re
        css_match = re.search(r'CUSTOM_CSS\s*=\s*"""(.*?)"""', content, re.DOTALL)
        if not css_match:
            print(f"   ✗ 未找到 CUSTOM_CSS")
            return False
        
        custom_css = css_match.group(1)
        
        # 检查 CSS 是否包含关键样式
        key_styles = [
            "--primary",
            "--bg",
            "--card",
            "gradio-container",
            "gradio-card",
            "gradio-button"
        ]
        
        found_styles = []
        for style in key_styles:
            if style in custom_css:
                found_styles.append(style)
                print(f"   ✓ 找到样式: {style}")
        
        if len(found_styles) == len(key_styles):
            print(f"   ✓ 所有关键样式都存在")
            print(f"   ✓ CSS 长度: {len(custom_css)} 字符")
            return True
        else:
            missing = set(key_styles) - set(found_styles)
            print(f"   ⚠️  缺少样式: {missing}")
            return False
        
    except Exception as e:
        print(f"   ✗ CSS 样式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("Gradio 前端应用测试程序")
    print("="*60)
    
    all_results = []
    
    # 测试依赖模块
    dependency_results = test_dependencies()
    all_results.extend(dependency_results)
    
    # 测试模块导入
    import_results = test_imports()
    all_results.extend(import_results)
    
    # 测试辅助函数
    helper_results = test_helper_functions()
    all_results.extend(helper_results)
    
    # 测试 CSS 样式
    css_ok = test_css_styles()
    all_results.append(("css_styles", css_ok))
    
    # 测试应用创建
    app_ok = test_app_creation()
    if app_ok is not None:
        all_results.append(("app_creation", app_ok))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in all_results:
        if result is None:
            status = "⚠ 跳过"
            skipped += 1
        elif result:
            status = "✓ 通过"
            passed += 1
        else:
            status = "✗ 失败"
            failed += 1
        print(f"{name:40s} : {status}")
    
    print(f"\n总计: {passed} 个通过, {failed} 个失败, {skipped} 个跳过")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
        print("\n启动方式：")
        print("  python frontend/start_gradio.py")
        print("  或")
        print("  python frontend/app.py")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
        print("\n注意：")
        print("1. 确保已安装 gradio: pip install gradio")
        print("2. 确保已安装 requests: pip install requests")
        print("3. 确保后端服务已启动（可选，用于完整功能测试）")
        return 1


if __name__ == "__main__":
    sys.exit(main())

