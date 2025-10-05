document.addEventListener('DOMContentLoaded', function() {
    const subcategoryCards = document.querySelectorAll('.subcategory-card');
    
    subcategoryCards.forEach(card => {
        card.addEventListener('click', function() {
            // Remove active class from all cards
            subcategoryCards.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked card
            this.classList.add('active');
            
            const category = this.dataset.category;
            const subcategory = this.dataset.subcategory;
            const url = this.dataset.url;
            
            console.log(`Selected: ${category} > ${subcategory}`);
            
            // Navigate to the URL in a new tab
            if (url) {
                window.open(url, '_blank');
            } else {
                console.warn(`No URL specified for ${category} > ${subcategory}`);
            }
        });
    });
});