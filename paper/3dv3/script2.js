document.addEventListener('DOMContentLoaded', function () {
    // 确保页面加载完毕时透明度为1
    document.body.style.opacity = '1';
});

document.querySelectorAll('.animate').forEach(item => {
    item.addEventListener('mouseover', function() {
        const previewDiv = this.nextElementSibling; // 获取动态图容器
        previewDiv.style.display = 'block'; // 显示动态图
    });
    item.addEventListener('mouseout', function() {
        const previewDiv = this.nextElementSibling; // 获取动态图容器
        previewDiv.style.display = 'none'; // 隐藏动态图
    });
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