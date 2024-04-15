document.addEventListener('DOMContentLoaded', function () {
    // 确保页面加载完毕时透明度为1
    document.body.style.opacity = '1';
});

// document.querySelectorAll('.animate').forEach(item => {
//     item.addEventListener('click', function(e) {
//         e.preventDefault();
//         const destination = this.getAttribute('href');
//         document.body.style.opacity = '0'; // 先设置整个页面透明度为0
//         setTimeout(() => {
//             window.location.href = destination; // 500毫秒后跳转
//         }, 500);
//     });
// });