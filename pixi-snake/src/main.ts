import {Application, Point} from "pixi.js";
import {SnakeView} from "./snake-view.ts";
import {Snake, SnakeAction} from "./snake.ts";

(async () => {
    // Create a new application
    const app = new Application();

    // Initialize the application
    await app.init({ background: "#1099bb", resizeTo: window });

    // Append the application canvas to the document body
    document.getElementById("pixi-container")!.appendChild(app.canvas);

    const gridSize = new Point(5, 5);
    const snake = new Snake(gridSize);
    SnakeView.Initialize(app.stage, gridSize);
    SnakeView.Draw(snake.reset().collisions, snake.reset().foodPosition);

    const stepMs = 500;
    let accumulatedTime = 0;
    // Listen for animate update
    app.ticker.add((time) => {
        accumulatedTime += time.deltaMS;

        while (accumulatedTime >= stepMs) {
            accumulatedTime -= stepMs;

            const action = Math.floor(Math.random() * 3) as SnakeAction;
            const stepResult = snake.step(action);

            let observation = stepResult.observation;
            if (stepResult.terminated)
            {
                observation = snake.reset();
                accumulatedTime = -2000
            }

            SnakeView.Draw(observation.collisions, observation.foodPosition);
        }
    });
})();
