/**
 * OpenAI 格式处理器
 * 处理 /v1/chat/completions 请求，支持流式和非流式响应
 */

import { generateAssistantResponse, generateAssistantResponseNoStream } from '../../api/client.js';
import { generateRequestBody, prepareImageRequest } from '../../utils/utils.js';
import { buildOpenAIErrorPayload } from '../../utils/errors.js';
import logger from '../../utils/logger.js';
import config from '../../config/config.js';
import tokenManager from '../../auth/token_manager.js';
import {
  createResponseMeta,
  setStreamHeaders,
  createHeartbeat,
  getChunkObject,
  releaseChunkObject,
  writeStreamData,
  endStream,
  with429Retry
} from '../stream.js';

/**
 * 创建流式数据块
 * 支持 DeepSeek 格式的 reasoning_content
 * @param {string} id - 响应ID
 * @param {number} created - 创建时间戳
 * @param {string} model - 模型名称
 * @param {Object} delta - 增量内容
 * @param {string|null} finish_reason - 结束原因
 * @returns {Object}
 */
export const createStreamChunk = (id, created, model, delta, finish_reason = null) => {
  const chunk = getChunkObject();
  chunk.id = id;
  chunk.object = 'chat.completion.chunk';
  chunk.created = created;
  chunk.model = model;
  chunk.choices[0].delta = delta;
  chunk.choices[0].finish_reason = finish_reason;
  return chunk;
};

/**
 * 处理 OpenAI 格式的聊天请求
 * @param {Request} req - Express请求对象
 * @param {Response} res - Express响应对象
 */
export const handleOpenAIRequest = async (req, res) => {
  const { messages, model, stream = false, tools, ...params } = req.body;
  
  try {
    if (!messages) {
      return res.status(400).json({ error: 'messages is required' });
    }

    // 判别逻辑：是否为生图模型
    const isImageModel = model.includes('-image');

    //console.log(JSON.stringify(requestBody,null,2));
    const { id, created } = createResponseMeta();
    const maxRetries = Number(config.retryTimes || 0);
    const safeRetries = maxRetries > 0 ? Math.floor(maxRetries) : 0;
    
    if (stream) {
      setStreamHeaders(res);
      
      // 启动心跳，防止 Cloudflare 超时断连
      const heartbeatTimer = createHeartbeat(res);

      try {
        if (isImageModel) {
          const { content, usage } = await with429Retry(
            async () => {
              const token = await tokenManager.getToken();
              if (!token) throw { status: 500, message: 'No available token' };
              try {
                const requestBody = generateRequestBody(messages, model, params, tools, token);
                prepareImageRequest(requestBody);
                return await generateAssistantResponseNoStream(requestBody, token);
              } catch (err) {
                const tokenIndex = tokenManager.tokens.indexOf(token);
                if (tokenIndex !== -1) err.tokenIndex = tokenIndex;
                
                // 生图空回不重试，返回空结果
                if (err.message && err.message.includes('Empty')) {
                  return { content: '', usage: null };
                }
                throw err;
              }
            },
            safeRetries,
            'chat.stream.image '
          );
          writeStreamData(res, createStreamChunk(id, created, model, { content }));
          writeStreamData(res, { ...createStreamChunk(id, created, model, {}, 'stop'), usage });
        } else {
          let hasToolCall = false;
          let usageData = null;

          await with429Retry(
            async () => {
              const token = await tokenManager.getToken();
              if (!token) throw { status: 500, message: 'No available token' };
              
              try {
                const requestBody = generateRequestBody(messages, model, params, tools, token);
                
                let hasContent = false;
                await generateAssistantResponse(requestBody, token, (data) => {
                  if (data.content && data.content.length > 0) {
                    hasContent = true;
                  }

                  if (data.type === 'usage') {
                    usageData = data.usage;
                  } else if (data.type === 'reasoning') {
                    const delta = { reasoning_content: data.reasoning_content };
                    if (data.thoughtSignature && config.passSignatureToClient) {
                      delta.thoughtSignature = data.thoughtSignature;
                    }
                    writeStreamData(res, createStreamChunk(id, created, model, delta));
                  } else if (data.type === 'tool_calls') {
                    hasToolCall = true;
                    const toolCallsWithIndex = data.tool_calls.map((toolCall, index) => {
                      if (config.passSignatureToClient) {
                        return { index, ...toolCall };
                      } else {
                        const { thoughtSignature, ...rest } = toolCall;
                        return { index, ...rest };
                      }
                    });
                    const delta = { tool_calls: toolCallsWithIndex };
                    writeStreamData(res, createStreamChunk(id, created, model, delta));
                  } else {
                    const delta = { content: data.content };
                    writeStreamData(res, createStreamChunk(id, created, model, delta));
                  }
                });

                if (!hasContent && !hasToolCall) {
                  throw { status: 503, message: 'Empty Stream Response' };
                }
              } catch (err) {
                const tokenIndex = tokenManager.tokens.indexOf(token);
                if (tokenIndex !== -1) err.tokenIndex = tokenIndex;
                throw err;
              }
            },
            safeRetries,
            'chat.stream '
          );

          writeStreamData(res, { ...createStreamChunk(id, created, model, {}, hasToolCall ? 'tool_calls' : 'stop'), usage: usageData });
        }

        clearInterval(heartbeatTimer);
        endStream(res);
      } catch (error) {
        clearInterval(heartbeatTimer);
        throw error;
      }
    } else {
      // 非流式请求：设置较长超时，避免大模型响应超时
      req.setTimeout(0); // 禁用请求超时
      res.setTimeout(0); // 禁用响应超时
      
      const { content, reasoningContent, reasoningSignature, toolCalls, usage } = await with429Retry(
        async () => {
          const token = await tokenManager.getToken();
          if (!token) throw { status: 500, message: 'No available token' };
          
          try {
            const requestBody = generateRequestBody(messages, model, params, tools, token);
            if (isImageModel) prepareImageRequest(requestBody);
            return await generateAssistantResponseNoStream(requestBody, token);
          } catch (err) {
            const tokenIndex = tokenManager.tokens.indexOf(token);
            if (tokenIndex !== -1) err.tokenIndex = tokenIndex;
            
            // 生图空回不重试，返回空结果
            if (isImageModel && err.message && err.message.includes('Empty')) {
              return { content: '', reasoningContent: null, reasoningSignature: null, toolCalls: [], usage: null };
            }
            throw err;
          }
        },
        safeRetries,
        'chat.no_stream '
      );
      
      // DeepSeek 格式：reasoning_content 在 content 之前
      const message = { role: 'assistant' };
      if (reasoningContent) message.reasoning_content = reasoningContent;
      if (reasoningSignature && config.passSignatureToClient) message.thoughtSignature = reasoningSignature;
      message.content = content;
      
      if (toolCalls.length > 0) {
        // 根据配置决定是否透传工具调用中的签名
        if (config.passSignatureToClient) {
          message.tool_calls = toolCalls;
        } else {
          message.tool_calls = toolCalls.map(({ thoughtSignature, ...rest }) => rest);
        }
      }
      
      // 使用预构建的响应对象，减少内存分配
      const response = {
        id,
        object: 'chat.completion',
        created,
        model,
        choices: [{
          index: 0,
          message,
          finish_reason: toolCalls.length > 0 ? 'tool_calls' : 'stop'
        }],
        usage
      };
      
      res.json(response);
    }
  } catch (error) {
    logger.error('生成响应失败:', error.message);
    if (res.headersSent) return;
    const statusCode = error.statusCode || error.status || 500;
    return res.status(statusCode).json(buildOpenAIErrorPayload(error, statusCode));
  }
};
