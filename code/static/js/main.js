document.addEventListener('DOMContentLoaded', function() {
    initRainParticles();
    initDragDrop();
    initImagePreview();
    initFormSubmission();
});

function initRainParticles() {
    const particlesContainer = document.createElement('div');
    particlesContainer.className = 'rain-particles';
    document.body.appendChild(particlesContainer);

    const particleCount = 30;

    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';

        particle.style.left = Math.random() * 100 + '%';

        const duration = (Math.random() * 3 + 2) + 's';
        particle.style.animationDuration = duration;

        particle.style.animationDelay = Math.random() * 5 + 's';

        particle.style.opacity = Math.random() * 0.5 + 0.2;

        particlesContainer.appendChild(particle);
    }
}

function initDragDrop() {
    const uploadZone = document.querySelector('.upload-zone');
    const fileInput = document.querySelector('input[type="file"]');

    if (!uploadZone || !fileInput) return;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.remove('drag-over');
        }, false);
    });

    uploadZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect(files[0]);
        }
    }, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function initImagePreview() {
    const fileInput = document.querySelector('input[type="file"]');

    if (!fileInput) return;

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file);
        }
    });
}

function handleFileSelect(file) {
    const validTypes = ['image/tiff', 'image/tif', 'image/png', 'image/jpeg', 'image/jpg'];
    const fileName = file.name.toLowerCase();
    const isValidExtension = fileName.endsWith('.tif') || fileName.endsWith('.tiff') ||
                            fileName.endsWith('.png') || fileName.endsWith('.jpg') ||
                            fileName.endsWith('.jpeg');

    if (!isValidExtension && !validTypes.includes(file.type)) {
        alert('Vui lòng chọn file ảnh hợp lệ (.TIF, .TIFF, .PNG, .JPG)');
        return;
    }

    const uploadText = document.querySelector('.upload-text');
    if (uploadText) {
        uploadText.innerHTML = `<i class="bi bi-check-circle"></i> ${file.name}`;
    }

    if (file.type.startsWith('image/') && !fileName.endsWith('.tif') && !fileName.endsWith('.tiff')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            showImagePreview(e.target.result);
        };
        reader.readAsDataURL(file);
    } else {
        const uploadZone = document.querySelector('.upload-zone');
        const existingPreview = uploadZone.querySelector('.image-preview');
        if (existingPreview) {
            existingPreview.remove();
        }
    }
}

function showImagePreview(dataUrl) {
    const uploadZone = document.querySelector('.upload-zone');

    const existingPreview = uploadZone.querySelector('.image-preview');
    if (existingPreview) {
        existingPreview.remove();
    }

    const preview = document.createElement('div');
    preview.className = 'image-preview';
    preview.innerHTML = `<img src="${dataUrl}" class="preview-img" alt="Preview">`;

    uploadZone.appendChild(preview);
}

function initFormSubmission() {
    const form = document.querySelector('form');
    const submitBtn = document.querySelector('.btn-primary');

    if (!form || !submitBtn) return;

    form.addEventListener('submit', (e) => {
        const fileInput = document.querySelector('input[type="file"]');

        if (!fileInput.files || fileInput.files.length === 0) {
            e.preventDefault();
            alert('Vui lòng chọn file ảnh trước khi dự đoán');
            return;
        }

        showLoading(submitBtn);
    });
}

function showLoading(btn) {
    const originalText = btn.innerHTML;
    btn.innerHTML = `
        <div class="loading-spinner" style="width: 20px; height: 20px; border-width: 2px; margin: 0;"></div>
        <span>Đang xử lý...</span>
    `;
    btn.disabled = true;
    btn.style.opacity = '0.7';
}

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

function animateRiskBadge() {
    const riskBadge = document.querySelector('.risk-badge');
    if (riskBadge) {
        riskBadge.style.animation = 'pulse 2s ease-in-out infinite';
    }
}

if (document.querySelector('.risk-badge')) {
    animateRiskBadge();
}

const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
`;
document.head.appendChild(style);
