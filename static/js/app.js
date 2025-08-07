// SimOpt Web GUI JavaScript

// Initialize app when DOM is ready
document.addEventListener("DOMContentLoaded", function () {
  initializeApp();
});

function initializeApp() {
  // Add loading indicators to HTMX requests
  document.body.addEventListener("htmx:beforeRequest", function (evt) {
    showLoadingState(evt.target);
  });

  document.body.addEventListener("htmx:afterRequest", function (evt) {
    hideLoadingState(evt.target);

    // Show success/error messages
    if (evt.detail.successful) {
      showSuccessMessage("Operation completed successfully");
    } else {
      showErrorMessage("Operation failed. Please try again.");
    }
  });

  // Add fade-in animation to new content
  document.body.addEventListener("htmx:afterSwap", function (evt) {
    evt.target.classList.add("fade-in");
    setTimeout(() => {
      evt.target.classList.remove("fade-in");
    }, 300);
  });

  // Initialize tooltips if Bootstrap is available
  if (typeof bootstrap !== "undefined") {
    var tooltipTriggerList = [].slice.call(
      document.querySelectorAll('[data-bs-toggle="tooltip"]')
    );
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
      return new bootstrap.Tooltip(tooltipTriggerEl);
    });
  }
}

function showLoadingState(element) {
  // Add loading indicator to buttons
  if (element.tagName === "BUTTON") {
    const originalHTML = element.innerHTML;
    element.setAttribute("data-original-html", originalHTML);
    element.innerHTML =
      '<span class="spinner-border spinner-border-sm me-2" role="status"></span>Loading...';
    element.disabled = true;
  }

  // Add loading class to containers
  element.classList.add("htmx-loading");
}

function hideLoadingState(element) {
  // Restore button state
  if (element.tagName === "BUTTON") {
    const originalHTML = element.getAttribute("data-original-html");
    if (originalHTML) {
      element.innerHTML = originalHTML;
      element.removeAttribute("data-original-html");
      element.disabled = false;
    }
  }

  // Remove loading class
  element.classList.remove("htmx-loading");
}

function showSuccessMessage(message) {
  showToast(message, "success");
}

function showErrorMessage(message) {
  showToast(message, "danger");
}

function showToast(message, type = "info") {
  // Create toast element
  const toastHTML = `
        <div class="toast align-items-center text-white bg-${type} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;

  // Get or create toast container
  let toastContainer = document.getElementById("toast-container");
  if (!toastContainer) {
    toastContainer = document.createElement("div");
    toastContainer.id = "toast-container";
    toastContainer.className = "toast-container position-fixed top-0 end-0 p-3";
    toastContainer.style.zIndex = "1100";
    document.body.appendChild(toastContainer);
  }

  // Add toast to container
  toastContainer.insertAdjacentHTML("beforeend", toastHTML);

  // Initialize and show toast if Bootstrap is available
  if (typeof bootstrap !== "undefined") {
    const toastElement = toastContainer.lastElementChild;
    const toast = new bootstrap.Toast(toastElement, {
      autohide: true,
      delay: 3000,
    });
    toast.show();

    // Remove toast element after it's hidden
    toastElement.addEventListener("hidden.bs.toast", function () {
      toastElement.remove();
    });
  }
}

// Utility functions
function confirmAction(message) {
  return confirm(message);
}

function validateForm(formElement) {
  const requiredFields = formElement.querySelectorAll("[required]");
  let isValid = true;

  requiredFields.forEach((field) => {
    if (!field.value.trim()) {
      field.classList.add("is-invalid");
      isValid = false;
    } else {
      field.classList.remove("is-invalid");
    }
  });

  return isValid;
}

// Export functions for global access
window.SimOptGUI = {
  showSuccessMessage,
  showErrorMessage,
  showToast,
  confirmAction,
  validateForm,
};

// Handle card hover effects on touch devices
document.addEventListener("touchstart", function () {
  document.body.classList.add("touch-device");
});

// Keyboard shortcuts (future enhancement)
document.addEventListener("keydown", function (e) {
  // Ctrl+N for new experiment
  if (e.ctrlKey && e.key === "n") {
    e.preventDefault();
    const newExpButton = document.querySelector('a[href="/experiments"]');
    if (newExpButton) newExpButton.click();
  }

  // Ctrl+H for home
  if (e.ctrlKey && e.key === "h") {
    e.preventDefault();
    const homeButton = document.querySelector('a[href="/"]');
    if (homeButton) homeButton.click();
  }
});

// Handle connection errors
document.body.addEventListener("htmx:responseError", function (evt) {
  console.error("HTMX Response Error:", evt.detail);
  showErrorMessage("Connection error. Please check your network connection.");
});

document.body.addEventListener("htmx:timeout", function (evt) {
  console.error("HTMX Timeout:", evt.detail);
  showErrorMessage("Request timed out. Please try again.");
});

// Auto-refresh functionality (optional)
function enableAutoRefresh(selector, interval = 30000) {
  const element = document.querySelector(selector);
  if (element && element.hasAttribute("hx-get")) {
    setInterval(() => {
      htmx.trigger(element, "refresh");
    }, interval);
  }
}

// Accessibility improvements
document.addEventListener("keydown", function (e) {
  // Skip links for keyboard navigation
  if (e.key === "Tab" && e.target.classList.contains("skip-link")) {
    e.preventDefault();
    const targetId = e.target.getAttribute("href").substring(1);
    const targetElement = document.getElementById(targetId);
    if (targetElement) {
      targetElement.focus();
    }
  }
});
