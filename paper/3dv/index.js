import * as SPLAT from "https://cdn.jsdelivr.net/npm/gsplat@latest";

const canvas = document.getElementById("canvas");
const progressDialog = document.getElementById("progress-dialog");
const progressIndicator = document.getElementById("progress-indicator");

const renderer = new SPLAT.WebGLRenderer(canvas);
const scene = new SPLAT.Scene();
const camera = new SPLAT.Camera();
const controls = new SPLAT.OrbitControls(camera, canvas);

async function main() {
    // Load and convert ply from url
    // const url =
    //     "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/bonsai/point_cloud/iteration_7000/point_cloud.ply";
    // const url = "model/door.ply"
    // const url = "model/LB.ply"
    // const url = "model/point_cloud.ply"
    // const url = "https://huggingface.co/datasets/Cfcys/gsplat_view/blob/main/LB.ply"
    const url = "https://huggingface.co/datasets/Cfcys/gsplat_view/resolve/main/LB.ply"
    await SPLAT.PLYLoader.LoadAsync(url, scene, (progress) => (progressIndicator.value = progress * 100));
    progressDialog.close();
    scene.saveToFile("bonsai.splat");

    // Render loop
    const frame = () => {
        controls.update();
        renderer.render(scene, camera);

        requestAnimationFrame(frame);
    };

    requestAnimationFrame(frame);
}

main();
