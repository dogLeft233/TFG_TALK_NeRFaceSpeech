import torch
import io
import base64
import logging
from typing import Optional, Dict, Any
from chatterbox.vc import ChatterboxVC
import soundfile as sf
import numpy as np
import os

# 配置日志
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VC_MODEL = None  # 延迟加载

#---------------------------------------------------------------------

class VCError(Exception):
    """自定义音色转换异常类"""
    def __init__(self, message: str, error_code: str = None, original_error: Exception = None):
        self.message = message
        self.error_code = error_code
        self.original_error = original_error
        super().__init__(self.message)
        
#---------------------------------------------------------------------

def load_vc_model():
    """延迟加载音色转换模型"""
    global VC_MODEL
    if VC_MODEL is None:
        try:
            logger.info(f"正在加载音色转换模型到设备: {DEVICE}")
            VC_MODEL = ChatterboxVC.from_pretrained(DEVICE)
            logger.info("音色转换模型加载成功")
        except Exception as e:
            error_msg = f"音色转换模型加载失败: {str(e)}"
            logger.error(error_msg)
            raise VCError(error_msg, "MODEL_LOAD_ERROR", e)
    return VC_MODEL

def unload_vc_model():
    """
    从内存中释放音色转换模型
    
    Returns:
        bool: 是否成功释放
    """
    global VC_MODEL
    try:
        if VC_MODEL is not None:
            logger.info("正在释放音色转换模型...")
            
            # 如果模型有清理方法，调用它
            if hasattr(VC_MODEL, 'cleanup'):
                VC_MODEL.cleanup()
            elif hasattr(VC_MODEL, 'close'):
                VC_MODEL.close()
            
            # 删除模型引用
            del VC_MODEL
            VC_MODEL = None
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 如果使用CUDA，清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("音色转换模型已成功释放")
            return True
        else:
            logger.info("音色转换模型未加载，无需释放")
            return True
            
    except Exception as e:
        error_msg = f"音色转换模型释放失败: {str(e)}"
        logger.error(error_msg)
        # 即使释放失败，也尝试重置全局变量
        VC_MODEL = None
        return False

def reload_vc_model():
    """
    重新加载音色转换模型（先释放再加载）
    
    Returns:
        bool: 是否成功重新加载
    """
    try:
        logger.info("开始重新加载音色转换模型...")
        
        # 先释放现有模型
        unload_success = unload_vc_model()
        if not unload_success:
            logger.warning("模型释放失败，但继续尝试重新加载")
        
        # 重新加载模型
        load_vc_model()
        
        logger.info("音色转换模型重新加载成功")
        return True
        
    except Exception as e:
        error_msg = f"音色转换模型重新加载失败: {str(e)}"
        logger.error(error_msg)
        raise VCError(error_msg, "MODEL_RELOAD_ERROR", e)

def get_vc_model_status():
    """
    获取音色转换模型状态
    
    Returns:
        dict: 模型状态信息
    """
    global VC_MODEL
    
    status = {
        'loaded': VC_MODEL is not None,
        'device': DEVICE,
        'cuda_available': torch.cuda.is_available(),
        'memory_info': {}
    }
    
    if VC_MODEL is not None:
        status['model_type'] = type(VC_MODEL).__name__
        
        # 获取内存使用情况
        if torch.cuda.is_available():
            status['memory_info'] = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated(),
                'max_reserved': torch.cuda.max_memory_reserved()
            }
    
    return status

#---------------------------------------------------------------------

def convert_voice_chatterbox(source_audio_path: str,
                            target_voice_path: str,
                            output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    使用chatterbox进行音色转换
    
    Args:
        source_audio_path: 源音频文件路径（要转换的音频）
        target_voice_path: 目标音色参考音频文件路径
        output_path: 输出文件路径（可选，如果不提供则返回内存中的音频数据）
    
    Returns:
        Dict[str, Any]: 包含转换后音频数据的字典
        {
            'success': bool,
            'data': {
                'wav_data': bytes,  # WAV文件的二进制数据（如果output_path为None）
                'base64_data': str,  # Base64编码的音频数据（如果output_path为None）
                'sample_rate': int,  # 采样率
                'duration': float,  # 音频时长（秒）
                'output_path': str,  # 输出文件路径（如果提供了output_path）
                'audio_info': dict  # 音频信息
            },
            'error': None or dict
        }
    """
    try:
        # 输入验证
        if not source_audio_path or not isinstance(source_audio_path, str):
            raise VCError("源音频路径不能为空且必须是字符串", "INVALID_INPUT")
        
        if not target_voice_path or not isinstance(target_voice_path, str):
            raise VCError("目标音色参考音频路径不能为空且必须是字符串", "INVALID_INPUT")
        
        # 检查文件是否存在
        if not os.path.exists(source_audio_path):
            raise VCError(f"源音频文件不存在: {source_audio_path}", "FILE_NOT_FOUND")
        
        if not os.path.exists(target_voice_path):
            raise VCError(f"目标音色参考音频文件不存在: {target_voice_path}", "FILE_NOT_FOUND")
        
        logger.info(f"开始音色转换")
        logger.debug(f"源音频: {source_audio_path}")
        logger.debug(f"目标音色参考: {target_voice_path}")
        logger.debug(f"输出路径: {output_path}")
        
        # 加载模型
        model = load_vc_model()
        
        # 执行音色转换
        logger.info("正在执行音色转换...")
        wav = model.generate(
            audio=source_audio_path,
            target_voice_path=target_voice_path
        )
        
        # 获取采样率
        if hasattr(model, 'sr'):
            sample_rate = model.sr
        elif hasattr(model, 'sample_rate'):
            sample_rate = model.sample_rate
        else:
            # 默认采样率
            sample_rate = 24000
            logger.warning(f"无法获取模型采样率，使用默认值: {sample_rate}Hz")
        
        # 验证生成的音频
        if wav is None:
            raise VCError("音色转换模型返回空音频", "EMPTY_AUDIO")
        
        # 转换为numpy数组
        if isinstance(wav, torch.Tensor):
            wav_np = wav.cpu().numpy()
        else:
            wav_np = np.array(wav)
        
        # 确保音频数据是1D数组
        if wav_np.ndim > 1:
            wav_np = wav_np.flatten()
            
        # 如果是 float，确保范围合理（-1..1）
        if np.issubdtype(wav_np.dtype, np.floating):
            max_abs = np.max(np.abs(wav_np)) if wav_np.size else 0.0
            if max_abs > 1.0:
                wav_np = wav_np / max_abs
        
        # 计算音频时长
        duration = len(wav_np) / sample_rate
        
        # 如果提供了输出路径，保存到文件
        if output_path and output_path.strip():  # 确保路径不为空且不是空白字符串
            try:
                # 获取目录路径，如果为空则跳过创建目录（文件在当前目录）
                output_dir = os.path.dirname(output_path)
                if output_dir:  # 只有当目录路径不为空时才创建
                    os.makedirs(output_dir, exist_ok=True)
                sf.write(output_path, wav_np, sample_rate, format='WAV')
                logger.info(f"音频已保存到: {output_path}")
            except Exception as e:
                logger.warning(f"保存音频文件失败: {str(e)}，将返回内存数据")
                output_path = None
        
        # 创建内存中的WAV文件（如果output_path为None或保存失败）
        wav_data = None
        base64_data = None
        if output_path is None:
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, wav_np, sample_rate, format='WAV')
            wav_data = wav_buffer.getvalue()
            wav_buffer.close()
            
            # 转换为Base64编码
            base64_data = base64.b64encode(wav_data).decode('utf-8')
        
        # 音频信息
        audio_info = {
            'sample_rate': sample_rate,
            'duration': duration,
            'channels': 1,
            'samples': len(wav_np),
            'format': 'WAV',
            'bit_depth': 16
        }
        
        logger.info(f"音色转换成功，音频时长: {duration:.2f}秒")
        
        return {
            'success': True,
            'data': {
                'wav_data': wav_data,
                'base64_data': base64_data,
                'sample_rate': sample_rate,
                'duration': duration,
                'output_path': output_path,
                'audio_info': audio_info
            },
            'error': None
        }
        
    except VCError:
        # 重新抛出VC错误
        raise
    except Exception as e:
        # 捕获其他异常并转换为VC错误
        error_msg = f"音色转换时发生未知错误: {str(e)}"
        logger.error(error_msg)
        raise VCError(error_msg, "VC_CONVERSION_ERROR", e)

#---------------------------------------------------------------------
    
def get_vc_response_api(source_audio_path: str,
                       target_voice_path: str,
                       output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    为前端提供的音色转换API接口，返回标准化的响应格式
    
    Args:
        source_audio_path: 源音频文件路径
        target_voice_path: 目标音色参考音频文件路径
        output_path: 输出文件路径（可选）
    
    Returns:
        Dict[str, Any]: 标准化的API响应
    """
    try:
        result = convert_voice_chatterbox(source_audio_path, target_voice_path, output_path)
        return result
    except VCError as e:
        return {
            'success': False,
            'data': None,
            'error': {
                'code': e.error_code,
                'message': e.message,
                'type': 'VCError'
            }
        }
    except Exception as e:
        return {
            'success': False,
            'data': None,
            'error': {
                'code': 'UNKNOWN_ERROR',
                'message': f"未知错误: {str(e)}",
                'type': 'Exception'
            }
        }

def save_wav_to_file(wav_data: bytes, file_path: str) -> bool:
    """
    将WAV数据保存到文件
    
    Args:
        wav_data: WAV二进制数据
        file_path: 保存路径
    
    Returns:
        bool: 是否保存成功
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(wav_data)
        logger.info(f"WAV文件已保存到: {file_path}")
        return True
    except Exception as e:
        logger.error(f"保存WAV文件失败: {str(e)}")
        return False

def manage_vc_model(action: str) -> Dict[str, Any]:
    """
    管理音色转换模型的API接口
    
    Args:
        action: 操作类型 ('load', 'unload', 'reload', 'status')
    
    Returns:
        Dict[str, Any]: 操作结果
    """
    try:
        if action == 'load':
            load_vc_model()
            return {
                'success': True,
                'message': '音色转换模型加载成功',
                'action': 'load'
            }
            
        elif action == 'unload':
            success = unload_vc_model()
            return {
                'success': success,
                'message': '音色转换模型释放成功' if success else '音色转换模型释放失败',
                'action': 'unload'
            }
            
        elif action == 'reload':
            reload_vc_model()
            return {
                'success': True,
                'message': '音色转换模型重新加载成功',
                'action': 'reload'
            }
            
        elif action == 'status':
            status = get_vc_model_status()
            return {
                'success': True,
                'data': status,
                'action': 'status'
            }
            
        else:
            return {
                'success': False,
                'error': {
                    'code': 'INVALID_ACTION',
                    'message': f'无效的操作: {action}。支持的操作: load, unload, reload, status',
                    'type': 'ValueError'
                }
            }
            
    except VCError as e:
        return {
            'success': False,
            'error': {
                'code': e.error_code,
                'message': e.message,
                'type': 'VCError'
            }
        }
    except Exception as e:
        return {
            'success': False,
            'error': {
                'code': 'UNKNOWN_ERROR',
                'message': f'未知错误: {str(e)}',
                'type': 'Exception'
            }
        }
    
#---------------------------------------------------------------------

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    import argparse
    
    parser = argparse.ArgumentParser(description='音色转换测试')
    parser.add_argument('--source_audio', type=str, required=True, help='源音频文件路径')
    parser.add_argument('--target_voice', type=str, required=True, help='目标音色参考音频文件路径')
    parser.add_argument('--output', type=str, default='vc_output.wav', help='输出文件路径')
    args = parser.parse_args()
    
    print("=== 音色转换测试开始 ===")
    try:
        # 测试1: 模型状态检查
        print("\n--- 测试1: 模型状态检查 ---")
        status_result = manage_vc_model('status')
        if status_result['success']:
            print(f"✅ 模型状态: {status_result['data']}")
        else:
            print(f"❌ 状态检查失败: {status_result['error']['message']}")
        
        # 测试2: 音色转换
        print("\n--- 测试2: 音色转换 ---")
        result = get_vc_response_api(
            source_audio_path=args.source_audio,
            target_voice_path=args.target_voice,
            output_path=args.output
        )
        
        if result['success']:
            print(f"✅ 音色转换成功")
            print(f"🎵 音频时长: {result['data']['duration']:.2f}秒")
            print(f"📊 采样率: {result['data']['sample_rate']}Hz")
            print(f"📁 音频信息: {result['data']['audio_info']}")
            if result['data']['output_path']:
                print(f"💾 输出文件: {result['data']['output_path']}")
            if result['data']['wav_data']:
                print(f"💾 WAV数据大小: {len(result['data']['wav_data'])} bytes")
                print(f"🔤 Base64数据长度: {len(result['data']['base64_data'])} 字符")
        else:
            print(f"❌ 音色转换失败: {result['error']['message']}")
        
        # 测试3: 模型释放
        print("\n--- 测试3: 模型释放 ---")
        unload_result = manage_vc_model('unload')
        if unload_result['success']:
            print(f"✅ {unload_result['message']}")
        else:
            print(f"❌ 模型释放失败: {unload_result['error']['message']}")
        
        # 测试4: 模型重新加载
        print("\n--- 测试4: 模型重新加载 ---")
        reload_result = manage_vc_model('reload')
        if reload_result['success']:
            print(f"✅ {reload_result['message']}")
        else:
            print(f"❌ 模型重新加载失败: {reload_result['error']['message']}")
        
        # 测试5: 重新检查状态
        print("\n--- 测试5: 重新检查状态 ---")
        final_status = manage_vc_model('status')
        if final_status['success']:
            print(f"✅ 最终模型状态: {final_status['data']}")
        else:
            print(f"❌ 状态检查失败: {final_status['error']['message']}")
            
    except Exception as e:
        print(f"💥 测试过程中发生异常: {str(e)}")
    
    print("\n=== 音色转换测试结束 ===")

