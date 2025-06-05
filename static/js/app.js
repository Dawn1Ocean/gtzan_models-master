// DOM 元素
const dropArea = document.getElementById('dropArea');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const currentFileName = document.getElementById('currentFileName');
const playBtn = document.getElementById('playBtn');
const pauseBtn = document.getElementById('pauseBtn');
const progressBar = document.getElementById('progressBar');
const currentTimeDisplay = document.getElementById('currentTime');
const durationDisplay = document.getElementById('duration');
const waveformCanvas = document.getElementById('waveformCanvas');
const spectrogramCanvas = document.getElementById('spectrogramCanvas');
const playhead = document.getElementById('playhead');
const genreAura = document.getElementById('genreAura');
const genreList = document.getElementById('genreList');
const genreChart = document.getElementById('genreChart');
const loadingOverlay = document.getElementById('loadingOverlay');
const fluidBackground = document.getElementById('fluid-background');

// 全局变量
let audioContext;
let audioBuffer;
let audioSource;
let analyser;
let dataArray;
let animationFrameId;
let fluidAnimationFrameId;
let isPlaying = false;
let startTime;
let waveformCtx = waveformCanvas.getContext('2d');
let spectrogramCtx = spectrogramCanvas.getContext('2d');
let uploadedFiles = [];
let activeFileIndex = -1;
let genreChartInstance = null;
let lastDrawTime = 0;
const FRAME_RATE = 60;
const FRAME_INTERVAL = 1000 / FRAME_RATE;
let currentGenreColors = []; // 当前流派颜色

// Audio analysis variables
const FFT_SIZE = 2048;
const SMOOTHING_TIME_CONSTANT = 0.85;

// Visualization variables
let energyHistory = new Array(100).fill(0);
let peakEnergy = 0;
let currentEnergy = 0;
let glowIntensity = 0;
let pulseScale = 1;

// 流派和颜色配置
const genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'];
const colors = {
    'blues': '#0033cc',     // 深蓝色
    'classical': '#e6b800', // 金色
    'country': '#996633',   // 棕色
    'disco': '#cc00cc',     // 紫色
    'hiphop': '#ff3300',    // 橙红色
    'jazz': '#009999',      // 青色
    'metal': '#333333',     // 深灰色
    'pop': '#ff66b3',       // 粉色
    'reggae': '#00cc00',    // 绿色
    'rock': '#cc0000'       // 红色
};

// 初始化
function init() {
    setupEventListeners();
    setupCanvasSize();
    window.addEventListener('resize', setupCanvasSize);
    
    // 初始化默认流体背景
    createDefaultFluidBackground();
    
    try {
        window.AudioContext = window.AudioContext || window.webkitAudioContext;
        audioContext = new AudioContext();
    } catch (e) {
        alert('Web Audio API 不受支持。请使用现代浏览器。');
    }
}

// 设置事件监听器
function setupEventListeners() {
    // 文件拖放
    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropArea.classList.add('drag-over');
    });

    dropArea.addEventListener('dragleave', () => {
        dropArea.classList.remove('drag-over');
    });

    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.classList.remove('drag-over');

        if (e.dataTransfer.files.length > 0) {
            handleFiles(e.dataTransfer.files);
        }
    });

    // 文件输入
    dropArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFiles(fileInput.files);
        }
    });

    // 音频控制
    playBtn.addEventListener('click', playAudio);
    pauseBtn.addEventListener('click', pauseAudio);

    // 进度条拖拽和点击
    const progressContainer = document.querySelector('.progress-container');

    // 进度条点击
    progressContainer.addEventListener('click', (e) => {
        if (audioBuffer && uploadedFiles[activeFileIndex].status === 'complete') {
            const rect = e.target.getBoundingClientRect();
            const clickPosition = (e.clientX - rect.left) / rect.width;
            seekAudio(clickPosition * audioBuffer.duration);
        }
    });

    // 进度条拖拽功能
    let isDragging = false;

    progressContainer.addEventListener('mousedown', (e) => {
        if (audioBuffer && uploadedFiles[activeFileIndex].status === 'complete') {
            isDragging = true;
            document.body.style.cursor = 'grabbing';

            // 禁用过渡效果以使拖动更流畅
            progressBar.style.transition = 'none';

            // 立即更新位置
            const rect = progressContainer.getBoundingClientRect();
            const clickPosition = (e.clientX - rect.left) / rect.width;
            progressBar.style.width = `${clickPosition * 100}%`;
            updatePlayhead(clickPosition);
        }
    });

    document.addEventListener('mousemove', (e) => {
        if (isDragging && audioBuffer) {
            const rect = progressContainer.getBoundingClientRect();
            let movePosition = (e.clientX - rect.left) / rect.width;

            // 限制在范围内
            movePosition = Math.max(0, Math.min(1, movePosition));

            // 更新UI
            progressBar.style.width = `${movePosition * 100}%`;
            updatePlayhead(movePosition);
            currentTimeDisplay.textContent = formatTime(movePosition * audioBuffer.duration);
        }
    });

    document.addEventListener('mouseup', (e) => {
        if (isDragging && audioBuffer) {
            isDragging = false;
            document.body.style.cursor = '';

            // 恢复过渡效果
            progressBar.style.transition = 'width 0.1s linear';

            // 计算最终位置并跳转
            const rect = progressContainer.getBoundingClientRect();
            let finalPosition = (e.clientX - rect.left) / rect.width;

            // 限制在范围内
            finalPosition = Math.max(0, Math.min(1, finalPosition));

            seekAudio(finalPosition * audioBuffer.duration);
        }
    });
}

// 设置Canvas大小
function setupCanvasSize() {
    const setCanvasSize = (canvas) => {
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
    };

    setCanvasSize(waveformCanvas);
    setCanvasSize(spectrogramCanvas);

    // 如果有活动文件，重新绘制可视化
    if (activeFileIndex !== -1 && uploadedFiles[activeFileIndex].audioData) {
        drawWaveform(uploadedFiles[activeFileIndex].audioData);
        drawSpectrogram(uploadedFiles[activeFileIndex].melSpectrogram);
    }
}

// 处理上传的文件
function handleFiles(files) {
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (file.type.startsWith('audio/')) {
            const fileIndex = uploadedFiles.length;

            // 添加到上传列表
            uploadedFiles.push({
                file: file,
                status: 'pending',
                predictions: null,
                audioData: null,
                melSpectrogram: null,
                currentTime: 0,
                progressWidth: 0
            });

            // 添加到UI
            const listItem = document.createElement('li');
            listItem.innerHTML = `
                <span class="file-name">${file.name}</span>
                <span class="file-status pending">等待处理</span>
            `;
            listItem.addEventListener('click', () => selectFile(fileIndex));
            fileList.appendChild(listItem);

            // 处理文件
            processFile(fileIndex);
        }
    }
}

// 处理文件（上传到服务器并获取预测结果）
function processFile(fileIndex) {
    const file = uploadedFiles[fileIndex].file;
    const formData = new FormData();
    formData.append('file', file);

    // 更新状态
    uploadedFiles[fileIndex].status = 'processing';
    updateFileStatus(fileIndex, 'processing', '处理中...');

    // 显示加载覆盖层
    loadingOverlay.style.display = 'flex';

    // 发送到服务器
    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('服务器响应错误');
            }
            return response.json();
        })
        .then(data => {
            // 保存结果
            uploadedFiles[fileIndex].status = 'complete';
            uploadedFiles[fileIndex].predictions = data.results.predictions;
            uploadedFiles[fileIndex].audioData = data.results.audio_data;
            uploadedFiles[fileIndex].melSpectrogram = data.results.mel_spectrogram;
            uploadedFiles[fileIndex].samplingRate = data.results.sampling_rate;

            // 更新UI
            updateFileStatus(fileIndex, 'complete', '完成');
            updateFileListGenre(fileIndex);

            // 如果这是第一个完成的文件，自动选择它
            if (activeFileIndex === -1) {
                selectFile(fileIndex);
            }

            // 隐藏加载覆盖层
            loadingOverlay.style.display = 'none';
        })
        .catch(error => {
            console.error('处理文件时出错:', error);
            uploadedFiles[fileIndex].status = 'error';
            updateFileStatus(fileIndex, 'error', '处理失败');
            loadingOverlay.style.display = 'none';
        });
}

// 更新文件状态UI
function updateFileStatus(fileIndex, statusClass, statusText) {
    const listItems = fileList.querySelectorAll('li');
    if (listItems[fileIndex]) {
        const statusElement = listItems[fileIndex].querySelector('.file-status');
        statusElement.className = `file-status ${statusClass}`;
        statusElement.textContent = statusText;
    }
}

// 选择文件
function selectFile(fileIndex) {
    if (isPlaying) {
        // 保存当前播放文件的进度
        const currentTime = audioContext.currentTime - startTime;
        if (activeFileIndex !== -1 && uploadedFiles[activeFileIndex]) {
            uploadedFiles[activeFileIndex].currentTime = currentTime;
            uploadedFiles[activeFileIndex].progressWidth = progressBar.style.width;
        }
        pauseAudio();
    }

    // 更新活动文件索引
    activeFileIndex = fileIndex;

    // 更新UI中的活动文件
    const listItems = fileList.querySelectorAll('li');
    listItems.forEach((item, index) => {
        if (index === fileIndex) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });

    const fileData = uploadedFiles[fileIndex];
    currentFileName.textContent = fileData.file.name;

    // 根据文件状态启用/禁用控件
    if (fileData.status === 'complete') {
        playBtn.disabled = false;
        pauseBtn.disabled = true;

        // 设置音频数据
        setAudioData(fileData.audioData, fileData.samplingRate);

        // 绘制可视化
        drawWaveform(fileData.audioData);
        drawSpectrogram(fileData.melSpectrogram);

        // 显示预测结果
        displayPredictions(fileData.predictions);

        // 恢复该文件的进度条和时间显示
        if (fileData.currentTime !== undefined) {
            // 恢复进度条
            progressBar.style.width = fileData.progressWidth || '0%';
            currentTimeDisplay.textContent = formatTime(fileData.currentTime);
            updatePlayhead(fileData.currentTime / audioBuffer.duration);
        } else {
            // 新文件，重置进度条
            progressBar.style.width = '0%';
            currentTimeDisplay.textContent = formatTime(0);
            updatePlayhead(0);
        }
    } else {
        playBtn.disabled = true;
        pauseBtn.disabled = true;
        resetVisualization();
    }
}

// 设置音频数据
function setAudioData(audioData, sampleRate) {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    // 如果音频上下文被挂起，恢复它
    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }

    // 创建音频分析器
    analyser = audioContext.createAnalyser();
    analyser.fftSize = FFT_SIZE;
    analyser.smoothingTimeConstant = SMOOTHING_TIME_CONSTANT;
    dataArray = new Uint8Array(analyser.frequencyBinCount);

    // 创建音频缓冲区
    const floatArray = new Float32Array(audioData);
    audioBuffer = audioContext.createBuffer(1, floatArray.length, sampleRate);
    audioBuffer.getChannelData(0).set(floatArray);

    // 更新持续时间显示
    updateDurationDisplay();

    // 重置能量历史
    energyHistory = new Array(100).fill(0);
    peakEnergy = 0;
    currentEnergy = 0;
}

// 更新持续时间显示
function updateDurationDisplay() {
    if (audioBuffer) {
        durationDisplay.textContent = formatTime(audioBuffer.duration);
    }
}

// 播放音频
function playAudio() {
    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }

    // 如果已经在播放，先停止
    if (isPlaying) {
        stopAudio();
    }

    // 创建新的音频源
    audioSource = audioContext.createBufferSource();
    audioSource.buffer = audioBuffer;

    // 连接分析器
    audioSource.connect(analyser);
    analyser.connect(audioContext.destination);

    // 获取当前文件的存储时间或从显示中获取
    let offset = 0;
    if (uploadedFiles[activeFileIndex] && uploadedFiles[activeFileIndex].currentTime !== undefined) {
        offset = uploadedFiles[activeFileIndex].currentTime;
    } else {
        offset = parseFloat(currentTimeDisplay.textContent.split(':').reduce((acc, time) => (60 * acc) + parseFloat(time)));
    }

    // 播放
    audioSource.start(0, offset);
    startTime = audioContext.currentTime - offset;
    isPlaying = true;

    // 更新UI
    playBtn.disabled = true;
    pauseBtn.disabled = false;
    playhead.style.display = 'block';

    // 启动更新循环
    updatePlayback();
    updateVisualization();
}

// 暂停音频
function pauseAudio() {
    if (isPlaying) {
        stopAudio();
        // 保持当前时间
        const currentTime = audioContext.currentTime - startTime;
        currentTimeDisplay.textContent = formatTime(currentTime);

        // 保存当前进度到文件对象
        if (activeFileIndex !== -1 && uploadedFiles[activeFileIndex]) {
            uploadedFiles[activeFileIndex].currentTime = currentTime;
            uploadedFiles[activeFileIndex].progressWidth = progressBar.style.width;
        }

        // 更新UI
        playBtn.disabled = false;
        pauseBtn.disabled = true;
    }
}

// 停止音频
function stopAudio() {
    if (audioSource) {
        audioSource.stop();
        audioSource.disconnect();
    }

    isPlaying = false;
    cancelAnimationFrame(animationFrameId);
}

// 跳转到特定时间
function seekAudio(time) {
    if (time < 0) time = 0;
    if (time > audioBuffer.duration) time = audioBuffer.duration;

    currentTimeDisplay.textContent = formatTime(time);

    // 更新进度条
    const progress = time / audioBuffer.duration;
    progressBar.style.width = `${progress * 100}%`;

    // 保存进度到当前文件
    if (activeFileIndex !== -1 && uploadedFiles[activeFileIndex]) {
        uploadedFiles[activeFileIndex].currentTime = time;
        uploadedFiles[activeFileIndex].progressWidth = `${progress * 100}%`;
    }

    // 更新播放头位置
    updatePlayhead(progress);

    // 如果正在播放，从新的位置开始
    if (isPlaying) {
        stopAudio();
        startTime = audioContext.currentTime - time;
        playAudio();
    }
}

// 更新播放
function updatePlayback() {
    if (!isPlaying) return;

    const currentTime = audioContext.currentTime - startTime;

    // 如果到达结尾，停止播放并重置进度条
    if (currentTime >= audioBuffer.duration) {
        stopAudio();
        // 重置到开头
        currentTimeDisplay.textContent = formatTime(0);
        progressBar.style.width = '0%';
        updatePlayhead(0);

        // 重置当前文件的进度信息
        if (activeFileIndex !== -1 && uploadedFiles[activeFileIndex]) {
            uploadedFiles[activeFileIndex].currentTime = 0;
            uploadedFiles[activeFileIndex].progressWidth = '0%';
        }

        playBtn.disabled = false;
        pauseBtn.disabled = true;
        return;
    }

    // 更新时间显示
    currentTimeDisplay.textContent = formatTime(currentTime);

    // 更新进度条
    const progress = currentTime / audioBuffer.duration;
    progressBar.style.width = `${progress * 100}%`;

    // 实时保存当前进度到文件对象
    if (activeFileIndex !== -1 && uploadedFiles[activeFileIndex]) {
        uploadedFiles[activeFileIndex].currentTime = currentTime;
        uploadedFiles[activeFileIndex].progressWidth = `${progress * 100}%`;
    }

    // 更新播放头位置
    updatePlayhead(progress);

    // 继续更新
    animationFrameId = requestAnimationFrame(updatePlayback);
}

// 更新播放头位置
function updatePlayhead(progress) {
    const width = waveformCanvas.width;
    playhead.style.left = `${progress * width}px`;
}

// 绘制波形
function drawWaveform(audioData) {
    const canvas = waveformCanvas;
    const ctx = waveformCtx;
    const width = canvas.width;
    const height = canvas.height;

    // 清除画布
    ctx.clearRect(0, 0, width, height);

    if (!audioData || audioData.length === 0) return;

    const data = audioData;
    const step = Math.ceil(data.length / width);
    const amp = height / 2;

    ctx.beginPath();
    ctx.moveTo(0, amp);

    // 绘制波形
    for (let i = 0; i < width; i++) {
        const min = Math.min(...data.slice(i * step, (i + 1) * step));
        const max = Math.max(...data.slice(i * step, (i + 1) * step));

        // 使用渐变色
        const gradient = ctx.createLinearGradient(0, amp - max * amp, 0, amp - min * amp);
        gradient.addColorStop(0, 'rgba(3, 218, 198, 0.8)');
        gradient.addColorStop(1, 'rgba(98, 0, 238, 0.8)');

        ctx.strokeStyle = gradient;
        ctx.lineWidth = 2;

        ctx.beginPath();
        ctx.moveTo(i, amp - min * amp);
        ctx.lineTo(i, amp - max * amp);
        ctx.stroke();
    }
}

// 绘制频谱图
function drawSpectrogram(melSpectrogram) {
    const canvas = spectrogramCanvas;
    const ctx = spectrogramCtx;
    const width = canvas.width;
    const height = canvas.height;

    // 清除画布
    ctx.clearRect(0, 0, width, height);

    if (!melSpectrogram || melSpectrogram.length === 0) return;

    const data = melSpectrogram;
    const specHeight = data.length;
    const specWidth = data[0].length;

    // 计算比例因子
    const scaleX = width / specWidth;
    const scaleY = height / specHeight;

    // 找出最大值和最小值以进行归一化
    let min = Infinity;
    let max = -Infinity;

    for (let i = 0; i < specHeight; i++) {
        for (let j = 0; j < specWidth; j++) {
            min = Math.min(min, data[i][j]);
            max = Math.max(max, data[i][j]);
        }
    }

    // 绘制频谱图
    for (let i = 0; i < specHeight; i++) {
        for (let j = 0; j < specWidth; j++) {
            const value = (data[i][j] - min) / (max - min);

            // 创建颜色渐变（从蓝色到紫色到红色）
            const r = Math.round(255 * Math.min(1, value * 2));
            const g = Math.round(255 * Math.min(1, 2 - value * 2));
            const b = Math.round(255 * (value < 0.5 ? value * 2 : 1));

            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.8)`;

            // 绘制矩形
            ctx.fillRect(
                j * scaleX,
                height - (i + 1) * scaleY, // 翻转以使低频在底部
                scaleX,
                scaleY
            );
        }
    }
}

// 显示预测结果
function displayPredictions(predictions) {
    // 清空列表
    genreList.innerHTML = '';

    // 创建光晕效果
    createGenreAura(predictions);

    // 创建图表
    createGenreChart(predictions);

    // 创建流体背景
    createFluidBackground(predictions);

    // 添加到列表
    predictions.forEach(prediction => {
        const percentage = Math.round(prediction.probability * 100);
        const listItem = document.createElement('li');
        listItem.className = 'genre-item';
        listItem.innerHTML = `
            <div class="genre-color" style="background-color: ${prediction.color};"></div>
            <span class="genre-name">${prediction.genre}</span>
            <span class="genre-probability">${percentage}%</span>
        `;
        genreList.appendChild(listItem);
    });
}

// 创建流派光晕效果
function createGenreAura(predictions) {
    const top3 = predictions.slice(0, 3);
    const total = top3.reduce((sum, p) => sum + p.probability, 0);
    let accumulated = 0;
    const stops = top3.map(p => {
        const start = (accumulated / total) * 100;
        accumulated += p.probability;
        const end = (accumulated / total) * 100;
        return `${p.color} ${start.toFixed(1)}% ${end.toFixed(1)}%`;
    });
    const gradientCSS = `conic-gradient(from 0deg, ${stops.join(', ')})`;
    genreAura.style.background = gradientCSS;
    // overlay text remains static
    genreAura.innerHTML = `<div class="genre-aura-text">${predictions[0].genre}</div>`;
}

// 更新文件列表条目以显示流派和颜色
function updateFileListGenre(fileIndex) {
    const listItems = fileList.querySelectorAll('li');
    const li = listItems[fileIndex];
    const fileData = uploadedFiles[fileIndex];
    if (fileData.predictions) {
        const pred = fileData.predictions[0];
        li.innerHTML = `
            <span class="file-color" style="background-color: ${pred.color};"></span>
            <span class="file-name">${fileData.file.name}</span>
            <span class="file-genre">${pred.genre}</span>
            <span class="file-status complete">完成</span>
        `;
        li.addEventListener('click', () => selectFile(fileIndex));
    }
}

// 创建流派图表
function createGenreChart(predictions) {
    // 销毁之前的图表实例
    if (genreChartInstance) {
        genreChartInstance.destroy();
    }

    // 准备数据
    const labels = predictions.map(p => p.genre);
    const data = predictions.map(p => p.probability * 100);
    const backgroundColors = predictions.map(p => p.color);

    // 创建图表
    const ctx = genreChart.getContext('2d');
    genreChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: '流派匹配度 (%)',
                data: data,
                backgroundColor: backgroundColors,
                borderColor: 'rgba(255, 255, 255, 0.2)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `匹配度: ${context.raw.toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                }
            }
        }
    });
}

// 创建流体背景效果
function createFluidBackground(predictions) {
    if (!predictions || predictions.length === 0) return;

    // 清除现有的背景
    fluidBackground.innerHTML = '';
    cancelAnimationFrame(fluidAnimationFrameId);

    // 获取前三种流派的颜色
    const top3 = predictions.slice(0, 3);
    const total = top3.reduce((sum, p) => sum + p.probability, 0);
    currentGenreColors = top3.map(p => ({
        color: p.color,
        weight: p.probability / total
    }));

    // 创建更多流体斑点，增加视觉丰富度
    for (let i = 0; i < 8; i++) {
        const blob = document.createElement('div');
        blob.className = 'fluid-blob';
        
        // 根据权重随机选择颜色
        const colorIndex = weightedRandomIndex(currentGenreColors.map(c => c.weight));
        blob.style.backgroundColor = currentGenreColors[colorIndex].color;
        
        // 随机大小和位置，让大小范围更宽
        const size = Math.random() * 450 + 200;
        blob.style.width = `${size}px`;
        blob.style.height = `${size}px`;
        
        // 为每个斑点添加唯一属性，以便后续动画有差异
        blob.dataset.velocity = 0.5 + Math.random() * 1.5; // 移动速度因子
        blob.dataset.pulseIntensity = 0.8 + Math.random() * 0.4; // 脉动强度因子
        
        // 设置初始位置
        resetBlobPosition(blob);
        
        // 添加到背景
        fluidBackground.appendChild(blob);
    }

    // 开始动画
    animateFluidBackground();
}

// 根据权重随机选择索引
function weightedRandomIndex(weights) {
    const sum = weights.reduce((a, b) => a + b, 0);
    const r = Math.random() * sum;
    let acc = 0;
    for (let i = 0; i < weights.length; i++) {
        acc += weights[i];
        if (r <= acc) return i;
    }
    return 0;
}

// 重置斑点位置
function resetBlobPosition(blob) {
    const x = Math.random() * window.innerWidth;
    const y = Math.random() * window.innerHeight;
    blob.style.left = `${x}px`;
    blob.style.top = `${y}px`;
}

// 动画流体背景
function animateFluidBackground() {
    const blobs = fluidBackground.querySelectorAll('.fluid-blob');
    
    // 动画参数
    const baseIntensity = isPlaying ? 1.5 : 0.3; // 增加播放时的基础强度
    const energyFactor = isPlaying ? currentEnergy / (peakEnergy || 0.1) : 0.1;
    const pulseScale = 1 + (energyFactor * 0.4); // 增加脉动幅度
    
    // 创建音频能量驱动的方向感，让流动有更明显的方向
    const directionX = Math.sin(Date.now() / 5000) * energyFactor;
    const directionY = Math.cos(Date.now() / 5000) * energyFactor;
    
    blobs.forEach(blob => {
        // 获取该斑点的唯一速度因子
        const velocityFactor = parseFloat(blob.dataset.velocity) || 1;
        const pulseIntensity = parseFloat(blob.dataset.pulseIntensity) || 1;
        
        // 当前位置
        const currentX = parseFloat(blob.style.left);
        const currentY = parseFloat(blob.style.top);
        
        // 计算移动量 - 结合随机性和音频能量引导的方向
        const energyDrivenX = directionX * 8 * velocityFactor * baseIntensity;
        const energyDrivenY = directionY * 8 * velocityFactor * baseIntensity;
        const randomX = (Math.random() - 0.5) * 4 * baseIntensity;
        const randomY = (Math.random() - 0.5) * 4 * baseIntensity;
        
        // 结合随机移动和能量驱动的方向移动
        const deltaX = energyDrivenX + randomX;
        const deltaY = energyDrivenY + randomY;
        
        // 设置新位置
        let newX = currentX + deltaX;
        let newY = currentY + deltaY;
        
        // 边界检查
        if (newX < -300 || newX > window.innerWidth + 300 || newY < -300 || newY > window.innerHeight + 300) {
            resetBlobPosition(blob);
        } else {
            blob.style.left = `${newX}px`;
            blob.style.top = `${newY}px`;
        }
        
        // 根据音频能量进行视觉效果增强
        if (isPlaying) {
            // 缩放效果 - 增加不同斑点间的变化差异
            const individualScale = pulseScale * pulseIntensity;
            
            // 模糊度随能量变化
            const blurValue = 50 + Math.round(energyFactor * 30);
            
            // 不透明度随能量变化
            const opacityValue = 0.5 + (energyFactor * 0.3);
            
            // 应用多种视觉变化
            blob.style.transform = `scale(${individualScale})`;
            blob.style.filter = `blur(${blurValue}px)`;
            blob.style.opacity = opacityValue;
        } else {
            // 静态模式下还原默认效果
            blob.style.transform = 'scale(1)';
            blob.style.filter = 'blur(60px)';
            blob.style.opacity = '0.6';
        }
    });
    
    // 继续动画
    fluidAnimationFrameId = requestAnimationFrame(animateFluidBackground);
}

// 更新可视化
function updateVisualization() {
    if (!isPlaying) return;

    const now = Date.now();
    if (now - lastDrawTime < FRAME_INTERVAL) {
        animationFrameId = requestAnimationFrame(updateVisualization);
        return;
    }

    lastDrawTime = now;

    // 获取频域数据
    analyser.getByteFrequencyData(dataArray);

    // 计算当前能量
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i];
    }
    currentEnergy = sum / dataArray.length / 255;

    // 更新峰值能量
    if (currentEnergy > peakEnergy) {
        peakEnergy = currentEnergy;
    } else {
        // 缓慢降低峰值，以适应歌曲的动态范围
        peakEnergy = peakEnergy * 0.995 + currentEnergy * 0.005;
    }

    // 更新能量历史
    energyHistory.push(currentEnergy);
    energyHistory.shift();

    // 继续动画
    animationFrameId = requestAnimationFrame(updateVisualization);
}

// 重置可视化
function resetVisualization() {
    // 清除波形和频谱图
    waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    spectrogramCtx.clearRect(0, 0, spectrogramCanvas.width, spectrogramCanvas.height);

    // 重置进度条和播放头
    progressBar.style.width = '0';
    playhead.style.display = 'none';

    // 重置时间显示
    currentTimeDisplay.textContent = '0:00';
    durationDisplay.textContent = '0:00';

    // 清空预测结果
    genreList.innerHTML = '';
    genreAura.style.background = 'none';
    genreAura.innerHTML = '';

    // 销毁图表
    if (genreChartInstance) {
        genreChartInstance.destroy();
        genreChartInstance = null;
    }

    // 重置流体背景
    cancelAnimationFrame(fluidAnimationFrameId);
    // 创建默认流体背景 (使用应用默认色彩)
    createDefaultFluidBackground();

    // 如果是当前文件，重置其进度信息
    if (activeFileIndex !== -1 && uploadedFiles[activeFileIndex]) {
        uploadedFiles[activeFileIndex].currentTime = 0;
        uploadedFiles[activeFileIndex].progressWidth = '0%';
    }
}

// 创建默认流体背景
function createDefaultFluidBackground() {
    // 清除现有的背景
    fluidBackground.innerHTML = '';
    
    // 默认颜色 (应用主题色)
    const defaultColors = [
        { color: 'var(--primary-color)', weight: 0.4 },
        { color: 'var(--secondary-color)', weight: 0.4 },
        { color: '#7c4dff', weight: 0.2 }
    ];
    
    currentGenreColors = defaultColors;
    
    // 创建流体斑点
    for (let i = 0; i < 8; i++) {
        const blob = document.createElement('div');
        blob.className = 'fluid-blob';
        
        // 根据权重随机选择颜色
        const colorIndex = weightedRandomIndex(defaultColors.map(c => c.weight));
        blob.style.backgroundColor = defaultColors[colorIndex].color;
        
        // 随机大小和位置
        const size = Math.random() * 450 + 200;
        blob.style.width = `${size}px`;
        blob.style.height = `${size}px`;
        
        // 为每个斑点添加唯一属性，以便后续动画有差异
        blob.dataset.velocity = 0.5 + Math.random() * 1.5;
        blob.dataset.pulseIntensity = 0.8 + Math.random() * 0.4;
        
        // 设置初始位置
        resetBlobPosition(blob);
        
        // 添加到背景
        fluidBackground.appendChild(blob);
    }
    
    // 开始动画
    animateFluidBackground();
}

// 格式化时间（秒 -> MM:SS）
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

// 页面加载完成后初始化
window.addEventListener('DOMContentLoaded', init);
