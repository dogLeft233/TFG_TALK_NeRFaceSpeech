import logging
import re
import argparse
from llm import get_llm_response_api, LLMError
from tts import get_tts_response_api, TTSError, manage_tts_model
from typing import List, Dict, Any

# 配置日志
logger = logging.getLogger(__name__)

class TalkError(Exception):
    """自定义Talk异常类"""
    def __init__(self, message: str, error_code: str = None, original_error: Exception = None):
        self.message = message
        self.error_code = error_code
        self.original_error = original_error
        super().__init__(self.message)

#---------------------------------------------------------------------

def split_text_to_sentences(text: str) -> List[str]:
    """
    将文本分割成句子
    
    Args:
        text: 输入文本
    
    Returns:
        List[str]: 句子列表
    """
    try:
        if not text or not isinstance(text, str):
            return []
        
        # 清理文本
        text = text.strip()
        if not text:
            return []
        
        # 使用正则表达式查找所有句子（保留分隔符）
        # 匹配句子内容 + 句子结束标点
        # text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s。，？！]', '', text)
        
        pattern = r'[^。！？,.!?]+[。！？,.!?]+'
        sentences = re.findall(pattern, text)
        
        # 如果没有匹配到（可能文本末尾没有标点），添加剩余部分
        matched_text = ''.join(sentences)
        if len(matched_text) < len(text):
            remaining = text[len(matched_text):].strip()
            if remaining:
                sentences.append(remaining)
        
        # 过滤空句子
        result = [s.strip() for s in sentences if s.strip()]
        
        # 如果没有分割出句子，返回原文本
        if not result:
            result = [text]
        
        head_len = 10
        logger.info(f"文本分割完成，共 {len(result)} 个句子")
        for i, sent in enumerate(result):
            logger.debug(f"第{i + 1}个句子，长度:{len(sent)},前{head_len}个字符:{sent[:head_len]}")
        
        return result
        
    except Exception as e:
        logger.warning(f"文本分割失败，返回原文本: {str(e)}")
        return [text]

def generate_audio_for_sentences(sentences: List[str], 
                               language_id: str = 'zh',
                               **tts_kwargs) -> List[Dict[str, Any]]:
    """
    为句子列表生成音频
    
    Args:
        sentences: 句子列表
        language_id: 语言ID
        **tts_kwargs: TTS参数
    
    Returns:
        List[Dict[str, Any]]: 音频结果列表
    """
    audio_results = []
    
    try:
        logger.info(f"开始为 {len(sentences)} 个句子生成音频")
        
        for i, sentence in enumerate(sentences):
            logger.info(f"正在处理第 {i+1}/{len(sentences)} 个句子: {sentence[:50]}...")
            
            # 调用TTS API
            tts_result = get_tts_response_api(sentence, language_id=language_id, **tts_kwargs)
            
            if tts_result['success']:
                audio_results.append({
                    'sentence': sentence,
                    'sentence_index': i,
                    'audio_data': tts_result['data']['wav_data'],
                    'base64_data': tts_result['data']['base64_data'],
                    'duration': tts_result['data']['duration'],
                    'sample_rate': tts_result['data']['sample_rate'],
                    'success': True,
                    'error': None
                })
                logger.info(f"第 {i+1} 个句子音频生成成功，时长: {tts_result['data']['duration']:.2f}秒")
            else:
                audio_results.append({
                    'sentence': sentence,
                    'sentence_index': i,
                    'audio_data': None,
                    'base64_data': None,
                    'duration': 0,
                    'sample_rate': 0,
                    'success': False,
                    'error': tts_result['error']
                })
                logger.error(f"第 {i+1} 个句子音频生成失败: {tts_result['error']['message']}")
        
        successful_count = sum(1 for result in audio_results if result['success'])
        logger.info(f"音频生成完成，成功: {successful_count}/{len(sentences)}")
        
        return audio_results
        
    except Exception as e:
        error_msg = f"批量音频生成失败: {str(e)}"
        logger.error(error_msg)
        raise TalkError(error_msg, "AUDIO_GENERATION_ERROR", e)

def combine_audio_data(audio_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    合并多个音频数据
    
    Args:
        audio_results: 音频结果列表
    
    Returns:
        Dict[str, Any]: 合并后的音频信息
    """
    try:
        import numpy as np
        import soundfile as sf
        import io
        import base64
        
        # 过滤成功的音频结果
        successful_results = [result for result in audio_results if result['success']]
        
        if not successful_results:
            return {
                'combined_audio_data': None,
                'combined_base64_data': None,
                'total_duration': 0,
                'sample_rate': 0,
                'success': False,
                'error': '没有成功的音频数据可合并'
            }
        
        # 获取采样率（假设所有音频采样率相同）
        sample_rate = successful_results[0]['sample_rate']
        
        # 合并音频数据
        combined_audio = []
        total_duration = 0
        
        for result in successful_results:
            # 从WAV数据中提取音频
            wav_buffer = io.BytesIO(result['audio_data'])
            audio_data, sr = sf.read(wav_buffer)
            wav_buffer.close()
            
            # 确保采样率一致
            if sr != sample_rate:
                logger.warning(f"采样率不一致: {sr} vs {sample_rate}")
            
            combined_audio.append(audio_data)
            total_duration += result['duration']
        
        # 拼接音频
        if combined_audio:
            combined_audio_array = np.concatenate(combined_audio)
            
            # 创建合并后的WAV文件
            combined_wav_buffer = io.BytesIO()
            sf.write(combined_wav_buffer, combined_audio_array, sample_rate, format='WAV')
            combined_wav_data = combined_wav_buffer.getvalue()
            combined_wav_buffer.close()
            
            # 转换为Base64
            combined_base64_data = base64.b64encode(combined_wav_data).decode('utf-8')
            
            logger.info(f"音频合并完成，总时长: {total_duration:.2f}秒")
            
            return {
                'combined_audio_data': combined_wav_data,
                'combined_base64_data': combined_base64_data,
                'total_duration': total_duration,
                'sample_rate': sample_rate,
                'success': True,
                'error': None
            }
        else:
            return {
                'combined_audio_data': None,
                'combined_base64_data': None,
                'total_duration': 0,
                'sample_rate': 0,
                'success': False,
                'error': '没有有效的音频数据'
            }
            
    except Exception as e:
        error_msg = f"音频合并失败: {str(e)}"
        logger.error(error_msg)
        return {
            'combined_audio_data': None,
            'combined_base64_data': None,
            'total_duration': 0,
            'sample_rate': 0,
            'success': False,
            'error': error_msg
        }

def talk_with_audio(user_input: str, 
                   language_id: str = 'zh',
                   combine_audio: bool = True,
                   release_tts_model: bool = False,
                   split_sentences: bool = True,
                   **tts_kwargs) -> Dict[str, Any]:
    """
    完整的对话功能：用户输入 -> LLM回答 -> TTS音频生成
    
    Args:
        user_input: 用户输入文本
        language_id: TTS语言ID
        combine_audio: 是否合并音频
        release_tts_model: 是否在完成后释放TTS模型
        split_sentences: 是否分句处理
        **tts_kwargs: TTS参数
    
    Returns:
        Dict[str, Any]: 完整的对话结果
    """
    try:
        logger.info(f"开始处理用户输入: {user_input[:100]}...")
        
        # 步骤1: 调用LLM获取回答
        logger.info("步骤1: 调用LLM生成回答...")
        llm_result = get_llm_response_api(user_input)
        
        if not llm_result['success']:
            return {
                'success': False,
                'error': {
                    'code': 'LLM_ERROR',
                    'message': f"LLM调用失败: {llm_result['error']['message']}",
                    'type': 'LLMError'
                },
                'data': None
            }
        
        llm_answer = llm_result['data']['answer']
        logger.info(f"LLM回答生成成功，长度: {len(llm_answer)}")
        
        # 步骤2: 分句处理（可选）
        if split_sentences:
            logger.info("步骤2: 对回答进行分句...")
            sentences = split_text_to_sentences(llm_answer)
            logger.info(f"分句完成，共 {len(sentences)} 个句子")
        else:
            logger.info("步骤2: 跳过分句，直接使用完整回答...")
            sentences = [llm_answer]
            logger.info("使用完整回答作为单个句子")
        
        # 步骤3: 生成音频
        logger.info("步骤3: 生成TTS音频...")
        audio_results = generate_audio_for_sentences(sentences, language_id=language_id, **tts_kwargs)
        
        # 检查是否有成功的音频
        successful_audio = [result for result in audio_results if result['success']]
        if not successful_audio:
            return {
                'success': False,
                'error': {
                    'code': 'TTS_ERROR',
                    'message': '所有句子的TTS生成都失败了',
                    'type': 'TTSError'
                },
                'data': {
                    'llm_answer': llm_answer,
                    'sentences': sentences,
                    'audio_results': audio_results
                }
            }
        
        # 步骤4:
        combined_audio_info = None
        if combine_audio:
            logger.info("步骤4: 处理合并音频...")
            combined_audio_info = combine_audio_data(audio_results)
        
        # 步骤5: 释放TTS模型（可选）
        tts_model_released = False
        if release_tts_model:
            logger.info("步骤5: 释放TTS模型...")
            try:
                unload_result = manage_tts_model('unload')
                tts_model_released = unload_result['success']
                if tts_model_released:
                    logger.info("TTS模型已成功释放")
                else:
                    logger.warning(f"TTS模型释放失败: {unload_result.get('message', '未知错误')}")
            except Exception as e:
                logger.error(f"TTS模型释放过程中发生异常: {str(e)}")
                tts_model_released = False
        
        # 构建返回结果
        result = {
            'success': True,
            'error': None,
            'data': {
                'user_input': user_input,
                'llm_answer': llm_answer,
                'sentences': sentences,
                'audio_results': audio_results,
                'successful_audio_count': len(successful_audio),
                'total_audio_count': len(audio_results),
                'combined_audio': combined_audio_info,
                'processing_info': {
                    'total_sentences': len(sentences),
                    'successful_audio': len(successful_audio),
                    'failed_audio': len(audio_results) - len(successful_audio),
                    'combine_audio_enabled': combine_audio,
                    'tts_model_released': tts_model_released,
                    'release_tts_model_enabled': release_tts_model,
                    'split_sentences_enabled': split_sentences
                }
            }
        }
        
        logger.info("对话处理完成")
        return result
        
    except TalkError:
        # 重新抛出TalkError
        raise
    except Exception as e:
        error_msg = f"对话处理过程中发生未知错误: {str(e)}"
        logger.error(error_msg)
        raise TalkError(error_msg, "TALK_PROCESSING_ERROR", e)

def get_talk_response_api(user_input: str, 
                         language_id: str = 'zh',
                         combine_audio: bool = True,
                         release_tts_model: bool = False,
                         split_sentences: bool = True,
                         **kwargs) -> Dict[str, Any]:
    """
    为前端提供的Talk API接口
    
    Args:
        user_input: 用户输入
        language_id: TTS语言ID
        combine_audio: 是否合并音频
        release_tts_model: 是否在完成后释放TTS模型
        split_sentences: 是否分句处理
        **kwargs: 其他参数
    
    Returns:
        Dict[str, Any]: 标准化的API响应
    """
    try:
        result = talk_with_audio(user_input, language_id=language_id, combine_audio=combine_audio, release_tts_model=release_tts_model, split_sentences=split_sentences, **kwargs)
        return result
    except TalkError as e:
        return {
            'success': False,
            'error': {
                'code': e.error_code,
                'message': e.message,
                'type': 'TalkError'
            },
            'data': None
        }
    except Exception as e:
        return {
            'success': False,
            'error': {
                'code': 'UNKNOWN_ERROR',
                'message': f'未知错误: {str(e)}',
                'type': 'Exception'
            },
            'data': None
        }



#---------------------------------------------------------------------

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='输入文字→返回文字→生成音频')
    parser.add_argument('--input_text', type=str, required=True, help='必填项，输入文本')
    parser.add_argument('--split_sentences', action='store_true', default=True, help='是否需要分词')
    parser.add_argument('--audio_prompt_path', type=str, default="", help='参考音色')
    parser.add_argument('--output_path', type=str, default="talk_output.wav", help='输出文件名')
    args = parser.parse_args()
    
    print("=== Talk功能测试开始 ===")
    input_text = args.input_text
    split_sentences = args.split_sentences
    audio_prompt_path = args.audio_prompt_path
    output_path = args.output_path
    
    try:
        print(f"用户输入: {input_text}")
        print(f"分句处理: {'是' if split_sentences else '否'}")
            
        # 调用Talk API
        result = get_talk_response_api(
            input_text, 
            combine_audio=True, 
            release_tts_model=True,
            split_sentences=split_sentences,
            audio_prompt_path=audio_prompt_path
        )
            
        if result['success']:
            data = result['data']
            print(f"✅ 处理成功")
            print(f"📝 LLM回答: {data['llm_answer']}")
            print(f"📊 句子数量: {data['processing_info']['total_sentences']}")
            print(f"🎵 成功音频: {data['processing_info']['successful_audio']}")
            print(f"❌ 失败音频: {data['processing_info']['failed_audio']}")
            print(f"✂️ 分句处理: {'是' if data['processing_info']['split_sentences_enabled'] else '否'}")
            print(f"🧠 TTS模型释放: {'是' if data['processing_info']['tts_model_released'] else '否'}")
                
            if data['combined_audio'] and data['combined_audio']['success']:
                print(f"🔗 合并音频时长: {data['combined_audio']['total_duration']:.2f}秒")
                
            # 保存合并音频（如果存在）
            if data['combined_audio'] and data['combined_audio']['success']:
                from tts import save_wav_to_file
                filename = output_path
                if save_wav_to_file(data['combined_audio']['combined_audio_data'], filename):
                    print(f"💾 合并音频已保存到: {filename}")
                
        else:
            print(f"❌ 处理失败: {result['error']['message']}")
    
    except Exception as e:
        print(f"💥 测试过程中发生异常: {str(e)}")
    
    print("\n=== Talk功能测试结束 ===")
