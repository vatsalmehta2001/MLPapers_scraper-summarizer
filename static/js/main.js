// Main JavaScript for ML Papers Summarizer

document.addEventListener('DOMContentLoaded', function() {
    // Add loading state to fetch papers form
    const fetchPapersForm = document.querySelector('form[action*="fetch-papers"]');
    if (fetchPapersForm) {
        fetchPapersForm.addEventListener('submit', function() {
            const submitButton = this.querySelector('button[type="submit"]');
            submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Fetching...';
            submitButton.disabled = true;
            
            // Add a message about the process taking time
            const modalBody = this.querySelector('.modal-body');
            const processingAlert = document.createElement('div');
            processingAlert.className = 'alert alert-warning mt-3 fetching-animation';
            processingAlert.innerHTML = '<i class="fas fa-sync-alt fa-spin"></i> Processing papers and generating summaries. This may take several minutes. Please do not close this window.';
            modalBody.appendChild(processingAlert);
        });
    }
    
    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert:not(.alert-info)');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Enable responsive table for mobile
    const tables = document.querySelectorAll('table');
    tables.forEach(table => {
        if (!table.parentElement.classList.contains('table-responsive')) {
            const wrapper = document.createElement('div');
            wrapper.className = 'table-responsive';
            table.parentNode.insertBefore(wrapper, table);
            wrapper.appendChild(table);
        }
    });
}); 