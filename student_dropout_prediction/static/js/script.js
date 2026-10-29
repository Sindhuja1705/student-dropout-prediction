document.addEventListener('DOMContentLoaded', function () {
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach((input) => {
        input.addEventListener('blur', () => {
            if (input.value.trim() !== '') {
                input.classList.add('is-valid');
            }
        });
    });
});
