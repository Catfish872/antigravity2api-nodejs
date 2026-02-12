import { spawn } from 'child_process';
import { existsSync, chmodSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { platform, arch } from 'os';
import zlib from 'zlib';

const __dirname = dirname(fileURLToPath(import.meta.url));
const isPkg = typeof process.pkg !== 'undefined';

// ==================== 辅助函数 ====================
function decompressGzip(buffer) {
  return new Promise((resolve, reject) => {
    zlib.gunzip(buffer, (err, result) => {
      if (err) reject(err);
      else resolve(result);
    });
  });
}

// ==================== 新版核心逻辑 (源自 liuw1535) ====================
class FingerprintRequester {
  constructor(options = {}) {
    this.binDir = options.binPath || this._detectBinDir();
    this.binaryPath = options.executablePath || this._detectBinary();
    // 自动指向 bin 目录下的 tls_config.json
    this.configPath = join(this.binDir, 'tls_config.json'); 
    this.defaults = {
      timeout: 30,
      proxy: null,
    };
    this.activeProcesses = new Set();
  }

  _detectBinDir() {
    if (isPkg) {
      const exeDir = dirname(process.execPath);
      const exeBinDir = join(exeDir, 'bin');
      if (existsSync(exeBinDir)) return exeBinDir;
      const cwdBinDir = join(process.cwd(), 'bin');
      if (existsSync(cwdBinDir)) return cwdBinDir;
    }
    return join(__dirname, 'bin');
  }

  _detectBinary() {
    const platformMap = { win32: 'windows', linux: 'linux', android: 'android', darwin: 'linux' };
    const archMap = { x64: 'amd64', arm64: 'arm64' };
    const os = platformMap[platform()];
    const cpuArch = archMap[arch()];

    if (!os || !cpuArch) throw new Error(`Unsupported platform: ${platform()} ${arch()}`);

    const ext = platform() === 'win32' ? '.exe' : '';
    // 注意：这里指向新的二进制文件名
    const binaryName = `fingerprint_${os}_${cpuArch}${ext}`;
    const binaryPath = join(this.binDir, binaryName);

    if (!existsSync(binaryPath)) {
        // 尝试回退查找（防止有些环境只拷贝了文件没改结构）
        console.warn(`Binary not found at ${binaryPath}, checking fallback...`);
    }

    if (platform() !== 'win32') {
      try { chmodSync(binaryPath, 0o755); } catch (e) {}
    }
    return binaryPath;
  }

  async request(config) {
    const {
      method = 'GET', url, headers = {}, data = '',
      timeout, proxy, responseType = 'text',
      onDownloadProgress, validateStatus = (status) => status >= 200 && status < 300,
      signal, skipGzipDecompress = false,
    } = config;

    const requestPayload = {
      method: method.toUpperCase(),
      url, headers,
      body: typeof data === 'string' ? data : JSON.stringify(data),
      config_path: this.configPath, // 传递 tls_config.json 路径
    };

    const timeoutSec = timeout || this.defaults.timeout;
    if (timeoutSec) requestPayload.timeout = { connect: timeoutSec, read: timeoutSec };

    const proxyUrl = proxy || this.defaults.proxy;
    if (proxyUrl) {
      const proxyType = proxyUrl.startsWith('socks') ? 'socks5' : 'http';
      requestPayload.proxy = { enabled: true, type: proxyType, url: proxyUrl };
    }

    return new Promise((resolve, reject) => {
      const proc = spawn(this.binaryPath);
      this.activeProcesses.add(proc);
      
      let headersParsed = false;
      let responseHeaders = {};
      let responseStatus = 200;
      let responseStatusText = 'OK';
      let headerBuffer = null;
      let bodyChunks = [];
      let totalLoaded = 0;
      let stderrData = '';

      const timeoutId = setTimeout(() => {
        proc.kill();
        reject(new Error('Request timeout'));
      }, timeoutSec * 1000);

      if (signal) {
        signal.addEventListener('abort', () => {
          proc.kill();
          reject(new Error('Request aborted'));
        });
      }

      proc.stdout.on('data', (chunk) => {
        if (!headersParsed) {
          if (!headerBuffer) headerBuffer = chunk;
          else headerBuffer = Buffer.concat([headerBuffer, chunk]);

          const separator = Buffer.from('\r\n\r\n');
          const headerEndIndex = headerBuffer.indexOf(separator);

          if (headerEndIndex !== -1) {
            const headerPart = headerBuffer.slice(0, headerEndIndex).toString('utf8');
            const bodyPart = headerBuffer.slice(headerEndIndex + 4);

            const lines = headerPart.split('\r\n');
            const statusMatch = lines[0].match(/HTTP\/[\d.]+ (\d+) (.+)/);
            responseStatus = statusMatch ? parseInt(statusMatch[1]) : 200;
            responseStatusText = statusMatch ? statusMatch[2] : 'OK';

            for (let i = 1; i < lines.length; i++) {
              const [key, ...valueParts] = lines[i].split(': ');
              if (key) responseHeaders[key.toLowerCase()] = valueParts.join(': ');
            }

            headersParsed = true;
            headerBuffer = null;
            clearTimeout(timeoutId);

            if (bodyPart.length > 0) {
              bodyChunks.push(bodyPart);
              totalLoaded += bodyPart.length;
              if (onDownloadProgress) {
                onDownloadProgress({
                  loaded: totalLoaded,
                  total: parseInt(responseHeaders['content-length']) || 0,
                  chunk: bodyPart.toString('utf8'),
                  status: responseStatus,
                  headers: responseHeaders,
                });
              }
            }
          }
        } else {
          bodyChunks.push(chunk);
          totalLoaded += chunk.length;
          if (onDownloadProgress) {
            onDownloadProgress({
              loaded: totalLoaded,
              total: parseInt(responseHeaders['content-length']) || 0,
              chunk: chunk.toString('utf8'),
              status: responseStatus,
              headers: responseHeaders,
            });
          }
        }
      });

      proc.stderr.on('data', (data) => stderrData += data.toString());

      proc.on('close', async (code) => {
        clearTimeout(timeoutId);
        this.activeProcesses.delete(proc);
        if (code !== 0) {
             // 简单的错误处理
             return reject(new Error(stderrData || `Process exited with code ${code}`));
        }

        try {
          let bodyBuffer = Buffer.concat(bodyChunks);
          const contentEncoding = responseHeaders['content-encoding'] || '';
          const isGzipData = bodyBuffer.length >= 2 && bodyBuffer[0] === 0x1f && bodyBuffer[1] === 0x8b;
          
          if (!skipGzipDecompress && contentEncoding.toLowerCase().includes('gzip') && isGzipData) {
            bodyBuffer = await decompressGzip(bodyBuffer);
          }

          const body = bodyBuffer.toString('utf8');
          let parsedData = body;
          if (responseType === 'json') {
             try { parsedData = JSON.parse(body); } catch(e) {}
          }

          const response = {
            data: parsedData,
            status: responseStatus,
            statusText: responseStatusText,
            headers: responseHeaders,
            config,
          };
          
          if (!validateStatus(responseStatus)) {
             const err = new Error(`Request failed with status code ${responseStatus}`);
             err.response = response;
             return reject(err);
          }
          resolve(response);
        } catch (err) {
          reject(err);
        }
      });
      
      proc.stdin.write(JSON.stringify(requestPayload));
      proc.stdin.end();
    });
  }

  // ==================== 兼容旧版 AntigravityRequester 的接口 ====================

  async antigravity_fetch(url, options = {}) {
    const config = {
      method: options.method || 'GET',
      url,
      headers: options.headers || {},
      data: options.body || '',
      timeout: options.timeout_ms ? Math.ceil(options.timeout_ms / 1000) : 30,
      proxy: options.proxy,
      skipGzipDecompress: false,
    };

    const response = await this.request(config);
    
    return {
      ok: response.status >= 200 && response.status < 300,
      status: response.status,
      statusText: response.statusText,
      headers: new Map(Object.entries(response.headers)),
      url,
      redirected: false,
      _data: response.data,
      async text() {
        return typeof this._data === 'string' ? this._data : JSON.stringify(this._data);
      },
      async json() {
        return typeof this._data === 'string' ? JSON.parse(this._data) : this._data;
      },
      async buffer() {
        return Buffer.from(typeof this._data === 'string' ? this._data : JSON.stringify(this._data), 'utf8');
      }
    };
  }

  antigravity_fetchStream(url, options = {}) {
    const streamResponse = new StreamResponse();
    const config = {
      method: options.method || 'GET',
      url,
      headers: options.headers || {},
      data: options.body || '',
      timeout: options.timeout_ms ? Math.ceil(options.timeout_ms / 1000) : 30,
      proxy: options.proxy,
      skipGzipDecompress: true, // 流式响应通常由上层处理解压或直接转发
      onDownloadProgress: ({ chunk, status, headers }) => {
        if (!streamResponse._started) {
          streamResponse._started = true;
          streamResponse.status = status;
          if (headers) streamResponse.headers = new Map(Object.entries(headers));
          if (streamResponse._onStart) streamResponse._onStart({ status, headers: streamResponse.headers });
        }
        if (streamResponse._onData) streamResponse._onData(chunk);
        streamResponse.chunks.push(chunk);
      },
      validateStatus: (status) => {
        streamResponse.status = status;
        return true; 
      },
    };

    this.request(config)
      .then((response) => {
        streamResponse.headers = new Map(Object.entries(response.headers));
        streamResponse._ended = true;
        streamResponse._finalText = streamResponse.chunks.join('');
        if (streamResponse._onEnd) streamResponse._onEnd();
      })
      .catch((error) => {
        streamResponse._ended = true;
        streamResponse._error = error;
        if (streamResponse._onError) streamResponse._onError(error);
      });

    return streamResponse;
  }

  close() {
    this.activeProcesses.forEach(proc => proc.kill());
    this.activeProcesses.clear();
  }
}

// ==================== 辅助类：流式响应 ====================
class StreamResponse {
  constructor() {
    this.status = null;
    this.headers = new Map();
    this.chunks = [];
    this._onStart = null;
    this._onData = null;
    this._onEnd = null;
    this._onError = null;
    this._ended = false;
    this._error = null;
    this._started = false;
  }
  onStart(cb) { this._onStart = cb; return this; }
  onData(cb) { this._onData = cb; return this; }
  onEnd(cb) { this._onEnd = cb; return this; }
  onError(cb) { this._onError = cb; return this; }
}

// 默认导出该类，保持与旧版 client.js 的兼容性
export default FingerprintRequester;
