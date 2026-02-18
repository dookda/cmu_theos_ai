/**
 * Class Panel - manages class selection UI with pixel counts
 */
const ClassPanel = {
    classes: [],
    activeIndex: 1, // default to vegetation

    async init() {
        const res = await fetch('/api/config');
        const data = await res.json();
        this.classes = data.classes;
        this.render();
    },

    render() {
        const container = document.getElementById('class-list');
        container.innerHTML = '';

        this.classes.forEach((cls, i) => {
            const btn = document.createElement('button');
            btn.className = 'class-btn' + (i === this.activeIndex ? ' active' : '');
            btn.dataset.index = i;

            const [r, g, b] = cls.color;
            btn.innerHTML = `
                <span class="class-color" style="background: rgb(${r},${g},${b})"></span>
                <span class="class-name">${cls.name}</span>
                <span class="class-count" id="class-count-${i}">0</span>
                <span class="class-key">${i + 1}</span>
            `;

            btn.addEventListener('click', () => this.select(i));
            container.appendChild(btn);
        });
    },

    select(index) {
        this.activeIndex = index;
        document.querySelectorAll('.class-btn').forEach(btn => {
            btn.classList.toggle('active', parseInt(btn.dataset.index) === index);
        });
    },

    getColor(index) {
        if (index < this.classes.length) {
            return this.classes[index].color;
        }
        return [0, 0, 0];
    },

    updateCounts() {
        if (!Canvas.labelMask) return;
        const counts = Canvas.getClassCounts();
        const total = 512 * 512;

        this.classes.forEach((cls, i) => {
            const el = document.getElementById(`class-count-${i}`);
            if (!el) return;
            const count = counts[i] || 0;
            const pct = (count / total * 100).toFixed(1);
            el.textContent = pct + '%';
        });
    }
};
