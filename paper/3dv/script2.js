document.querySelectorAll('.animate').forEach(item => {
    item.addEventListener('click', function(e) {
        e.preventDefault();
        const destination = this.getAttribute('href');
        // 这里可以添加更复杂的动画
        document.querySelector('.container').style.opacity = '0';
        setTimeout(() => {
            window.location.href = destination;
        }, 500); // 延迟500毫秒后跳转，让动画效果可见
    });
});

document.addEventListener('DOMContentLoaded', function () {
    document.body.style.opacity = '1'; // 确保页面加载时不透明度为1
});

document.querySelectorAll('.animate').forEach(item => {
    item.addEventListener('click', function(e) {
        e.preventDefault();
        const destination = this.getAttribute('href');
        document.body.style.opacity = '0';
        setTimeout(() => {
            window.location.href = destination;
        }, 500); // 延迟500毫秒后跳转，让动画效果可见
    });
});

