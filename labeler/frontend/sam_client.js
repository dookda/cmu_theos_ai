/**
 * SAM Client - API calls for SAM inference
 */
const SAMClient = {
    isEmbedding: false,

    async embed(filename) {
        this.isEmbedding = true;
        try {
            const res = await fetch(`/api/sam/embed/${filename}`, { method: 'POST' });
            return await res.json();
        } finally {
            this.isEmbedding = false;
        }
    },

    async predict(filename, points, pointLabels, box) {
        const body = { filename };
        if (points && points.length > 0) {
            body.points = points;
            body.point_labels = pointLabels;
        }
        if (box) {
            body.box = box;
        }

        const res = await fetch('/api/sam/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        return await res.json();
    },

    async decodeMaskBase64(b64) {
        // Decode base64 PNG mask to Uint8Array (binary: 0 or 1)
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = 512;
                canvas.height = 512;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, 512, 512);
                const data = ctx.getImageData(0, 0, 512, 512).data;

                const mask = new Uint8Array(512 * 512);
                for (let i = 0; i < 512 * 512; i++) {
                    mask[i] = data[i * 4] > 127 ? 1 : 0;
                }
                resolve(mask);
            };
            img.src = `data:image/png;base64,${b64}`;
        });
    },

    async precompute() {
        const res = await fetch('/api/sam/precompute', { method: 'POST' });
        return await res.json();
    }
};
