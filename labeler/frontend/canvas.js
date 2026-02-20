/**
 * Canvas - handles image display, overlay rendering, drawing, zoom/pan, brightness/contrast
 */
const Canvas = {
    imageCanvas: null,
    overlayCanvas: null,
    interactionCanvas: null,
    imageCtx: null,
    overlayCtx: null,
    interCtx: null,

    labelMask: null,       // Uint8Array(512*512), values 0-6
    undoStack: [],
    maxUndo: 20,
    opacity: 0.5,
    brushSize: 10,
    isDirty: false,

    // SAM preview state
    samPreviewMask: null,
    samRawMask: null, // original SAM output before grow/shrink
    growRadius: 0,    // positive = dilate, negative = erode

    // Zoom/pan state
    zoom: 1,
    panX: 0,
    panY: 0,
    isPanning: false,
    panStartX: 0,
    panStartY: 0,

    // Image adjustment
    brightness: 100,
    contrast: 100,
    rawImage: null, // store original image for re-rendering

    init() {
        this.imageCanvas = document.getElementById('image-canvas');
        this.overlayCanvas = document.getElementById('overlay-canvas');
        this.interactionCanvas = document.getElementById('interaction-canvas');
        this.imageCtx = this.imageCanvas.getContext('2d');
        this.overlayCtx = this.overlayCanvas.getContext('2d');
        this.interCtx = this.interactionCanvas.getContext('2d');

        this.labelMask = new Uint8Array(512 * 512);
        this.undoStack = [];

        this.fitViewport();
        this.setupZoom();

        window.addEventListener('resize', () => this.fitViewport());
    },

    fitViewport() {
        const container = document.getElementById('canvas-container');
        const viewport = document.getElementById('canvas-viewport');
        // Available space: container minus padding (24px each side) and view-controls (~50px + gap)
        const availW = container.clientWidth - 48 - 56;
        const availH = container.clientHeight - 48 - 40; // bottom hint area
        const size = Math.max(320, Math.min(availW, availH));

        viewport.style.width = size + 'px';
        viewport.style.height = size + 'px';

        // Center the 512px canvas wrapper inside the viewport with base scale
        this.baseScale = size / 512;
        const wrapper = document.getElementById('canvas-wrapper');
        wrapper.style.left = '0px';
        wrapper.style.top = '0px';

        // Reset zoom/pan relative to new size
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.applyTransform();
    },

    setupZoom() {
        const viewport = document.getElementById('canvas-viewport');

        viewport.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? -0.1 : 0.1;
            const newZoom = Math.max(0.5, Math.min(5, this.zoom + delta));

            // Zoom toward mouse position
            const rect = viewport.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;

            // Adjust pan so zoom centers on mouse
            this.panX = mx - (mx - this.panX) * (newZoom / this.zoom);
            this.panY = my - (my - this.panY) * (newZoom / this.zoom);

            this.zoom = newZoom;
            this.applyTransform();
        }, { passive: false });

        // Middle-click pan
        viewport.addEventListener('mousedown', (e) => {
            if (e.button === 1) {
                e.preventDefault();
                this.isPanning = true;
                this.panStartX = e.clientX - this.panX;
                this.panStartY = e.clientY - this.panY;
                viewport.style.cursor = 'grabbing';
            }
        });

        window.addEventListener('mousemove', (e) => {
            if (this.isPanning) {
                this.panX = e.clientX - this.panStartX;
                this.panY = e.clientY - this.panStartY;
                this.applyTransform();
            }
        });

        window.addEventListener('mouseup', (e) => {
            if (e.button === 1 && this.isPanning) {
                this.isPanning = false;
                document.getElementById('canvas-viewport').style.cursor = '';
            }
        });

        // Prevent middle-click default
        viewport.addEventListener('auxclick', (e) => {
            if (e.button === 1) e.preventDefault();
        });

        // Two-finger pinch-to-zoom and pan
        let lastPinchDist = null;
        let lastPinchMidX = null;
        let lastPinchMidY = null;

        viewport.addEventListener('touchstart', (e) => {
            if (e.touches.length === 2) {
                e.preventDefault();
                const dx = e.touches[0].clientX - e.touches[1].clientX;
                const dy = e.touches[0].clientY - e.touches[1].clientY;
                lastPinchDist = Math.sqrt(dx * dx + dy * dy);
                lastPinchMidX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
                lastPinchMidY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
            }
        }, { passive: false });

        viewport.addEventListener('touchmove', (e) => {
            if (e.touches.length === 2 && lastPinchDist !== null) {
                e.preventDefault();
                const dx = e.touches[0].clientX - e.touches[1].clientX;
                const dy = e.touches[0].clientY - e.touches[1].clientY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
                const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
                const rect = viewport.getBoundingClientRect();

                // Zoom toward pinch midpoint
                const newZoom = Math.max(0.5, Math.min(5, this.zoom * (dist / lastPinchDist)));
                const mx = midX - rect.left;
                const my = midY - rect.top;
                this.panX = mx - (mx - this.panX) * (newZoom / this.zoom);
                this.panY = my - (my - this.panY) * (newZoom / this.zoom);

                // Pan with finger movement
                this.panX += midX - lastPinchMidX;
                this.panY += midY - lastPinchMidY;

                this.zoom = newZoom;
                lastPinchDist = dist;
                lastPinchMidX = midX;
                lastPinchMidY = midY;
                this.applyTransform();
            }
        }, { passive: false });

        viewport.addEventListener('touchend', () => {
            lastPinchDist = null;
            lastPinchMidX = null;
            lastPinchMidY = null;
        });
    },

    applyTransform() {
        const wrapper = document.getElementById('canvas-wrapper');
        const s = (this.baseScale || 1) * this.zoom;
        wrapper.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${s})`;
        document.getElementById('zoom-info').textContent = `${Math.round(this.zoom * 100)}%`;
    },

    resetZoom() {
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.fitViewport();
    },

    // Convert screen coords to canvas coords accounting for zoom/pan
    screenToCanvas(clientX, clientY) {
        const viewport = document.getElementById('canvas-viewport');
        const rect = viewport.getBoundingClientRect();
        const sx = clientX - rect.left;
        const sy = clientY - rect.top;
        const s = (this.baseScale || 1) * this.zoom;
        const x = (sx - this.panX) / s;
        const y = (sy - this.panY) / s;
        return { x: Math.floor(x), y: Math.floor(y) };
    },

    async loadImage(filename) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                this.rawImage = img;
                this.renderImage();
                resolve();
            };
            img.onerror = reject;
            img.src = `/api/tiles/${filename}`;
        });
    },

    renderImage() {
        if (!this.rawImage) return;
        this.imageCtx.filter = `brightness(${this.brightness}%) contrast(${this.contrast}%)`;
        this.imageCtx.drawImage(this.rawImage, 0, 0, 512, 512);
        this.imageCtx.filter = 'none';
    },

    setBrightness(val) {
        this.brightness = val;
        this.renderImage();
    },

    setContrast(val) {
        this.contrast = val;
        this.renderImage();
    },

    loadMask(maskArray) {
        this.labelMask = new Uint8Array(maskArray);
        this.isDirty = false;
        this.undoStack = [];
        this.renderOverlay();
    },

    clearMask() {
        this.pushUndo();
        this.labelMask.fill(0);
        this.isDirty = true;
        this.renderOverlay();
    },

    renderOverlay() {
        const imgData = this.overlayCtx.createImageData(512, 512);
        const data = imgData.data;
        const alpha = Math.round(this.opacity * 255);

        for (let i = 0; i < 512 * 512; i++) {
            const cls = this.labelMask[i];
            const px = i * 4;

            if (cls === 0) {
                data[px] = 0;
                data[px + 1] = 0;
                data[px + 2] = 0;
                data[px + 3] = 0;
            } else {
                const [r, g, b] = ClassPanel.getColor(cls);
                data[px] = r;
                data[px + 1] = g;
                data[px + 2] = b;
                data[px + 3] = alpha;
            }
        }

        this.overlayCtx.putImageData(imgData, 0, 0);
        ClassPanel.updateCounts();
    },

    renderSAMPreview(binaryMask) {
        this.samRawMask = new Uint8Array(binaryMask);
        this.applyGrowAndRender();
    },

    applyGrowAndRender() {
        if (!this.samRawMask) return;
        const mask = this.growRadius !== 0
            ? this.morphMask(this.samRawMask, this.growRadius)
            : this.samRawMask;
        this.samPreviewMask = mask;

        const imgData = this.interCtx.createImageData(512, 512);
        const data = imgData.data;
        const [r, g, b] = ClassPanel.getColor(ClassPanel.activeIndex);

        for (let i = 0; i < 512 * 512; i++) {
            const px = i * 4;
            if (mask[i]) {
                data[px] = r;
                data[px + 1] = g;
                data[px + 2] = b;
                data[px + 3] = 120;
            }
        }

        this.interCtx.putImageData(imgData, 0, 0);
    },

    // Morphological dilate (radius > 0) or erode (radius < 0) on a 512x512 binary mask
    morphMask(src, radius) {
        const w = 512, h = 512;
        const abs = Math.abs(radius);
        const isDilate = radius > 0;
        const out = new Uint8Array(w * h);
        const r2 = abs * abs;

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = y * w + x;
                if (isDilate) {
                    // Output 1 if any neighbor within radius is 1
                    let found = false;
                    for (let dy = -abs; dy <= abs && !found; dy++) {
                        for (let dx = -abs; dx <= abs && !found; dx++) {
                            if (dx * dx + dy * dy > r2) continue;
                            const nx = x + dx, ny = y + dy;
                            if (nx >= 0 && nx < w && ny >= 0 && ny < h && src[ny * w + nx]) {
                                found = true;
                            }
                        }
                    }
                    out[idx] = found ? 1 : 0;
                } else {
                    // Output 1 only if all neighbors within radius are 1
                    if (!src[idx]) { out[idx] = 0; continue; }
                    let allSet = true;
                    for (let dy = -abs; dy <= abs && allSet; dy++) {
                        for (let dx = -abs; dx <= abs && allSet; dx++) {
                            if (dx * dx + dy * dy > r2) continue;
                            const nx = x + dx, ny = y + dy;
                            if (nx < 0 || nx >= w || ny < 0 || ny >= h || !src[ny * w + nx]) {
                                allSet = false;
                            }
                        }
                    }
                    out[idx] = allSet ? 1 : 0;
                }
            }
        }
        return out;
    },

    setGrowRadius(val) {
        this.growRadius = val;
        if (this.samRawMask) {
            this.applyGrowAndRender();
        }
    },

    clearInteraction() {
        this.interCtx.clearRect(0, 0, 512, 512);
        this.samPreviewMask = null;
        this.samRawMask = null;
    },

    acceptSAMPreview() {
        if (!this.samPreviewMask) return false;

        this.pushUndo();
        const cls = ClassPanel.activeIndex;
        for (let i = 0; i < 512 * 512; i++) {
            if (this.samPreviewMask[i]) {
                this.labelMask[i] = cls;
            }
        }
        this.isDirty = true;
        this.clearInteraction();
        this.renderOverlay();
        return true;
    },

    paintAt(x, y, classIndex) {
        const r = this.brushSize;
        const r2 = r * r;

        for (let dy = -r; dy <= r; dy++) {
            for (let dx = -r; dx <= r; dx++) {
                if (dx * dx + dy * dy > r2) continue;
                const px = x + dx;
                const py = y + dy;
                if (px < 0 || px >= 512 || py < 0 || py >= 512) continue;
                this.labelMask[py * 512 + px] = classIndex;
            }
        }
        this.isDirty = true;
    },

    pushUndo() {
        this.undoStack.push(new Uint8Array(this.labelMask));
        if (this.undoStack.length > this.maxUndo) {
            this.undoStack.shift();
        }
    },

    undo() {
        if (this.undoStack.length === 0) return false;
        this.labelMask = this.undoStack.pop();
        this.isDirty = true;
        this.renderOverlay();
        return true;
    },

    getMaskAsBase64() {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(512, 512);

        for (let i = 0; i < 512 * 512; i++) {
            const v = this.labelMask[i];
            const px = i * 4;
            imgData.data[px] = v;
            imgData.data[px + 1] = v;
            imgData.data[px + 2] = v;
            imgData.data[px + 3] = 255;
        }

        ctx.putImageData(imgData, 0, 0);
        const dataUrl = canvas.toDataURL('image/png');
        return dataUrl.split(',')[1];
    },

    setOpacity(val) {
        this.opacity = val;
        this.renderOverlay();
    },

    drawPoint(x, y, isPositive) {
        this.interCtx.beginPath();
        this.interCtx.arc(x, y, 5, 0, Math.PI * 2);
        this.interCtx.fillStyle = isPositive ? '#4ade80' : '#ef4444';
        this.interCtx.fill();
        this.interCtx.strokeStyle = '#fff';
        this.interCtx.lineWidth = 2;
        this.interCtx.stroke();
    },

    drawBox(x1, y1, x2, y2) {
        this.interCtx.strokeStyle = '#4ade80';
        this.interCtx.lineWidth = 2;
        this.interCtx.setLineDash([5, 5]);
        this.interCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        this.interCtx.setLineDash([]);
    },

    getClassCounts() {
        const counts = {};
        for (let i = 0; i < 512 * 512; i++) {
            const cls = this.labelMask[i];
            counts[cls] = (counts[cls] || 0) + 1;
        }
        return counts;
    }
};
