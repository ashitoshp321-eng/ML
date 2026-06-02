/**
 * ML Testing Platform - Main Interactive Frontend Logic
 */

document.addEventListener('DOMContentLoaded', function() {
    // ----------------------------------------------------
    // Theme Management (Dark / Light Mode)
    // ----------------------------------------------------
    const themeToggler = document.getElementById('themeToggler');
    const htmlElement = document.documentElement;
    
    // Check local storage or system default
    const savedTheme = localStorage.getItem('theme');
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme) {
        htmlElement.setAttribute('data-theme', savedTheme);
        updateTogglerIcon(savedTheme);
    } else if (systemPrefersDark) {
        htmlElement.setAttribute('data-theme', 'dark');
        updateTogglerIcon('dark');
    } else {
        htmlElement.setAttribute('data-theme', 'light');
        updateTogglerIcon('light');
    }
    
    if (themeToggler) {
        themeToggler.addEventListener('click', function() {
            const currentTheme = htmlElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            
            htmlElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateTogglerIcon(newTheme);
            
            // Dispatch event for any active Plotly charts to redraw with new themes
            window.dispatchEvent(new Event('themeChanged'));
        });
    }
    
    function updateTogglerIcon(theme) {
        if (!themeToggler) return;
        const moonIcon = themeToggler.querySelector('.fa-moon');
        const sunIcon = themeToggler.querySelector('.fa-sun');
        
        if (theme === 'dark') {
            if (moonIcon) moonIcon.style.display = 'none';
            if (sunIcon) sunIcon.style.display = 'block';
        } else {
            if (moonIcon) moonIcon.style.display = 'block';
            if (sunIcon) sunIcon.style.display = 'none';
        }
    }

    // ----------------------------------------------------
    // Responsive Sidebar Toggle
    // ----------------------------------------------------
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('appSidebar');
    
    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', function(e) {
            e.stopPropagation();
            sidebar.classList.toggle('active');
        });
        
        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', function(e) {
            if (sidebar.classList.contains('active') && !sidebar.contains(e.target) && e.target !== sidebarToggle) {
                sidebar.classList.remove('active');
            }
        });
    }

    // ----------------------------------------------------
    // Loading Screen Helpers
    // ----------------------------------------------------
    const loadingOverlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');
    
    window.showLoading = function(text = "Processing Pipeline...") {
        if (loadingOverlay) {
            if (loadingText) loadingText.textContent = text;
            loadingOverlay.classList.add('active');
        }
    };
    
    window.hideLoading = function() {
        if (loadingOverlay) {
            loadingOverlay.classList.remove('active');
        }
    };

    // Attach to form submissions that require training or complex backend processing
    const trainingForms = document.querySelectorAll('.trigger-loading-form');
    trainingForms.forEach(form => {
        form.addEventListener('submit', function() {
            const loadingMsg = this.getAttribute('data-loading-msg') || "Training Model Pipeline...";
            window.showLoading(loadingMsg);
        });
    });

    // ----------------------------------------------------
    // Bootstrap Form Validation Enablement
    // ----------------------------------------------------
    const validationForms = document.querySelectorAll('.needs-validation');
    Array.from(validationForms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
});
