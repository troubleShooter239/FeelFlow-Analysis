document.addEventListener('DOMContentLoaded', function() {
    var signupBtn = document.getElementById('signupBtn');
    var closeBtn = document.getElementById('signupCloseBtn');
    
    var signupModal = document.getElementById('signupModal');
    

    signupBtn.addEventListener('click', function() {
        signupModal.style.display = 'block';
    });

    closeBtn.addEventListener('click', function() {
        signupModal.style.display = 'none';
    });

    window.addEventListener('click', function(event) {
        if (event.target == signupModal) {
            signupModal.style.display = 'none';
        }
    });
});