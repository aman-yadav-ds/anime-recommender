document.addEventListener("DOMContentLoaded", () => {
    // Get DOM elements
    const cards = document.querySelectorAll(".anime-card");
    const modalBackdrop = document.getElementById("modal-backdrop");
    const cardModal = document.getElementById("card-modal");
    const modalClose = document.getElementById("modal-close");
    const themeToggle = document.getElementById("theme-toggle");
    const themeIcon = document.querySelector(".theme-icon");
    const body = document.body;
    
    let isAnimating = false;

    // Initialize theme
    const savedTheme = localStorage.getItem("theme") || "light";
    document.documentElement.setAttribute("data-theme", savedTheme);
    updateThemeIcon(savedTheme);

    // Card click functionality
    cards.forEach((card, index) => {
        // Add staggered animation delay for initial load
        card.style.animationDelay = `${index * 0.1}s`;
        
        card.addEventListener("click", (e) => {
            // Don't expand if clicking on heart button
            if (e.target.closest(".heart-btn")) return;
            
            if (isAnimating) return;
            
            // Get card data from data attributes
            const img = card.querySelector("img");
            const title = card.getAttribute("data-title");
            const score = card.getAttribute("data-score");
            const genres = card.getAttribute("data-genres");
            const episodes = card.getAttribute("data-episodes");
            const premiered = card.getAttribute("data-premiered");
            const synopsis = card.getAttribute("data-synopsis");
            
            openModal({
                title: title,
                image: img.src,
                score: `⭐ ${score}/10`,
                genres: genres,
                episodes: episodes,
                premiered: premiered,
                synopsis: synopsis
            });
        });

        // Heart button functionality
        const heartBtn = card.querySelector(".heart-btn");
        if (heartBtn) {
            heartBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                toggleHeart(heartBtn);
            });
        }
    });

    // Close modal functionality
    if (modalClose) {
        modalClose.addEventListener("click", closeModal);
    }

    // Modal backdrop click to close
    if (modalBackdrop) {
        modalBackdrop.addEventListener("click", closeModal);
    }

    // Escape key to close modal
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            closeModal();
        }
    });

    // Theme toggle functionality
    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            const currentTheme = document.documentElement.getAttribute("data-theme");
            const newTheme = currentTheme === "dark" ? "light" : "dark";
            
            // Add transition class for smooth theme change
            document.documentElement.classList.add("theme-transition");
            document.documentElement.setAttribute("data-theme", newTheme);
            
            // Save theme preference
            localStorage.setItem("theme", newTheme);
            updateThemeIcon(newTheme);
            
            // Remove transition class after animation
            setTimeout(() => {
                document.documentElement.classList.remove("theme-transition");
            }, 300);
        });
    }

    // Error message fade out
    const errorBox = document.querySelector(".error-message");
    if (errorBox) {
        setTimeout(() => {
            errorBox.classList.add("fade-out");
            setTimeout(() => {
                errorBox.remove();
            }, 300);
        }, 4000);
    }

    // Smooth scroll animation for new recommendations
    const recommendations = document.querySelector(".recommendations");
    if (recommendations) {
        // Add entrance animation
        recommendations.style.opacity = "0";
        recommendations.style.transform = "translateY(30px)";
        
        setTimeout(() => {
            recommendations.style.transition = "all 0.6s cubic-bezier(0.4, 0, 0.2, 1)";
            recommendations.style.opacity = "1";
            recommendations.style.transform = "translateY(0)";
        }, 100);
    }

    // Add loading animation to form submission
    const form = document.querySelector(".input-form");
    if (form) {
        form.addEventListener("submit", (e) => {
            const submitBtn = form.querySelector(".submit-btn");
            const btnText = submitBtn.querySelector(".btn-text");
            const btnIcon = submitBtn.querySelector(".btn-icon");
            
            if (submitBtn && btnText) {
                submitBtn.disabled = true;
                submitBtn.style.transform = "scale(0.95)";
                btnText.textContent = "Searching...";
                if (btnIcon) btnIcon.textContent = "⏳";
                
                // Add pulsing animation
                submitBtn.style.animation = "pulse 1.5s ease-in-out infinite";
            }
        });
    }

    // Functions
    function openModal(data) {
        if (isAnimating) return;
        
        isAnimating = true;
        
        // Set modal data
        document.getElementById("modal-title").textContent = data.title;
        document.getElementById("modal-image").src = data.image;
        document.getElementById("modal-image").alt = data.title;
        document.getElementById("modal-score").textContent = data.score;
        document.getElementById("modal-genres").textContent = data.genres;
        document.getElementById("modal-episodes").textContent = data.episodes;
        document.getElementById("modal-premiered").textContent = data.premiered;
        document.getElementById("modal-synopsis").textContent = data.synopsis;
        
        // Show modal
        if (modalBackdrop) {
            modalBackdrop.classList.remove("hidden");
            modalBackdrop.classList.add("active");
        }
        
        if (cardModal) {
            cardModal.classList.remove("hidden");
            cardModal.classList.add("active");
        }
        
        body.classList.add("modal-open");
        
        setTimeout(() => {
            isAnimating = false;
        }, 300);
    }

    function closeModal() {
        if (isAnimating) return;
        
        isAnimating = true;
        
        // Hide modal
        if (cardModal) {
            cardModal.classList.remove("active");
        }
        
        if (modalBackdrop) {
            modalBackdrop.classList.remove("active");
        }
        
        body.classList.remove("modal-open");
        
        setTimeout(() => {
            if (cardModal) cardModal.classList.add("hidden");
            if (modalBackdrop) modalBackdrop.classList.add("hidden");
            isAnimating = false;
        }, 300);
    }

    function toggleHeart(heartBtn) {
        const heartIcon = heartBtn.querySelector(".heart-icon");
        const isActive = heartBtn.classList.contains("active");
        
        // Add click animation
        heartBtn.style.transform = "scale(0.8)";
        
        setTimeout(() => {
            if (isActive) {
                heartBtn.classList.remove("active");
                heartIcon.textContent = "🤍";
            } else {
                heartBtn.classList.add("active");
                heartIcon.textContent = "❤️";
                
                // Add floating heart animation
                createFloatingHeart(heartBtn);
            }
            
            heartBtn.style.transform = "scale(1.2)";
            
            setTimeout(() => {
                heartBtn.style.transform = "scale(1)";
            }, 150);
        }, 100);
    }

    function createFloatingHeart(button) {
        const heart = document.createElement("div");
        heart.textContent = "❤️";
        heart.style.cssText = `
            position: absolute;
            font-size: 20px;
            pointer-events: none;
            z-index: 1000;
            animation: floatHeart 1.5s ease-out forwards;
        `;
        
        const rect = button.getBoundingClientRect();
        heart.style.left = rect.left + rect.width / 2 + "px";
        heart.style.top = rect.top + rect.height / 2 + "px";
        
        document.body.appendChild(heart);
        
        setTimeout(() => {
            heart.remove();
        }, 1500);
    }

    function updateThemeIcon(theme) {
        if (themeIcon) {
            themeIcon.textContent = theme === "dark" ? "☀️" : "🌙";
        }
    }

    // Add intersection observer for card animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: "0px 0px -50px 0px"
    };

    const cardObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.style.animation = "slideInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards";
                cardObserver.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all cards for entrance animation
    cards.forEach((card, index) => {
        card.style.opacity = "0";
        card.style.transform = "translateY(30px)";
        card.style.animationDelay = `${index * 0.1}s`;
        cardObserver.observe(card);
    });

    // Add CSS animations dynamically
    const style = document.createElement("style");
    style.textContent = `
        @keyframes slideInUp {
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes modalFadeIn {
            from {
                opacity: 0;
                transform: translate(-50%, -50%) scale(0.8);
            }
            to {
                opacity: 1;
                transform: translate(-50%, -50%) scale(1);
            }
        }
        
        @keyframes modalFadeOut {
            from {
                opacity: 1;
                transform: translate(-50%, -50%) scale(1);
            }
            to {
                opacity: 0;
                transform: translate(-50%, -50%) scale(0.8);
            }
        }
        
        @keyframes floatHeart {
            0% {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
            100% {
                opacity: 0;
                transform: translateY(-50px) scale(1.5);
            }
        }
        
        .theme-transition * {
            transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease !important;
        }
        
        .anime-card {
            animation-fill-mode: both;
        }
    `;
    document.head.appendChild(style);

    // Add smooth scrolling behavior
    document.documentElement.style.scrollBehavior = "smooth";

    // Add loading states for images with error handling
    const images = document.querySelectorAll(".anime-card img");
    images.forEach(img => {
        // Set default styles
        img.style.opacity = "0";
        img.style.transition = "opacity 0.3s ease";
        
        // Handle successful image load
        img.addEventListener("load", () => {
            img.style.opacity = "1";
        });
        
        // Handle image load errors
        img.addEventListener("error", () => {
            // Create a placeholder div
            const placeholder = document.createElement("div");
            placeholder.style.cssText = `
                width: 100%;
                height: 100%;
                background: linear-gradient(135deg, #f3f4f6, #e5e7eb);
                display: flex;
                align-items: center;
                justify-content: center;
                color: #6b7280;
                font-weight: 600;
                text-align: center;
                padding: 20px;
            `;
            placeholder.innerHTML = "📺<br>No Image<br>Available";
            
            // Replace the image with placeholder
            img.parentNode.replaceChild(placeholder, img);
        });
        
        // Force load check for cached images
        if (img.complete) {
            img.style.opacity = "1";
        }
    });

    // Add performance optimization for animations
    let rafId = null;
    const throttledResize = () => {
        if (rafId) return;
        rafId = requestAnimationFrame(() => {
            // Handle responsive adjustments here if needed
            rafId = null;
        });
    };

    window.addEventListener("resize", throttledResize);

    // Add focus management for accessibility
    document.addEventListener("keydown", (e) => {
        if (e.key === "Tab" && expandedCard) {
            const focusableElements = expandedCard.querySelectorAll(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            
            if (focusableElements.length > 0) {
                const firstElement = focusableElements[0];
                const lastElement = focusableElements[focusableElements.length - 1];
                
                if (e.shiftKey && document.activeElement === firstElement) {
                    e.preventDefault();
                    lastElement.focus();
                } else if (!e.shiftKey && document.activeElement === lastElement) {
                    e.preventDefault();
                    firstElement.focus();
                }
            }
        }
    });
});