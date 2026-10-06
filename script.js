// Fade-in Animation
function initFadeInAnimations() {
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    const sections = document.querySelectorAll('.section, .hero-section, .citation-section, .footer');
    sections.forEach(section => {
        section.classList.add('fade-in-section');
        observer.observe(section);
    });
}

// Place the sidebar gradient indicator on the current page's link
function initSidebarIndicator() {
    const sidebar = document.querySelector('.sidebar');
    const activeLink = document.querySelector('.sidebar-item.active');
    if (!sidebar || !activeLink) return;

    function update() {
        const linkRect = activeLink.getBoundingClientRect();
        const sidebarRect = sidebar.getBoundingClientRect();
        sidebar.style.setProperty('--active-position', `${linkRect.top - sidebarRect.top - 32}px`);
    }

    window.addEventListener('resize', update, { passive: true });
    update();
}

function copyBibtex() {
    const text = document.getElementById('bibtex-text').innerText;
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.querySelector('.citation-copy-btn');
        const original = btn.innerHTML;
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
        setTimeout(() => { btn.innerHTML = original; }, 2000);
    });
}

// Image zoom functionality
function initImageZoom() {
    const images = document.querySelectorAll(".zoomable-image");
    if (!images.length) return;

    const zoomOverlay = document.createElement("div");
    zoomOverlay.classList.add("zoom-overlay");

    const zoomImage = document.createElement("img");
    zoomOverlay.appendChild(zoomImage);
    document.body.appendChild(zoomOverlay);

    images.forEach((img) => {
        // Preload high-res image if available
        const highResSrc = img.getAttribute("data-highres");
        if (highResSrc) {
            const preloadImg = new Image();
            preloadImg.src = highResSrc;
        }

        img.addEventListener("click", () => {
            zoomImage.src = highResSrc || img.src;
            zoomOverlay.classList.add("active");
        });
    });

    // Close the zoom overlay when clicking anywhere on it
    zoomOverlay.addEventListener("click", () => {
        zoomOverlay.classList.remove("active");
        zoomImage.src = "";
    });

    // Allow closing with the ESC key
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") zoomOverlay.classList.remove("active");
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initSidebarIndicator();
    initFadeInAnimations();
    initImageZoom();
});
