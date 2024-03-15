document.addEventListener('DOMContentLoaded', function() {
    var loginBtn = document.getElementById('loginBtn');
    var closeBtn = document.getElementById('loginCloseBtn');
    
    var loginModal = document.getElementById('loginModal');
    

    loginBtn.addEventListener('click', function() {
        loginModal.style.display = 'block';
    });

    closeBtn.addEventListener('click', function() {
        loginModal.style.display = 'none';
    });

    window.addEventListener('click', function(event) {
        if (event.target == loginModal) {
            loginModal.style.display = 'none';
        }
    });
});
