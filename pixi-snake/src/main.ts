import {Application, Point} from "pixi.js";
import {SnakeView} from "./snake-view.ts";
import {Snake} from "./snake.ts";
import {SnakeNet} from "./snake-net.ts";

(async () => {
    // Create a new application
    const app = new Application();

    // Initialize the application
    await app.init({ background: "#1099bb", resizeTo: window });

    // Append the application canvas to the document body
    document.getElementById("pixi-container")!.appendChild(app.canvas);

    const snakeNet = await SnakeNet.create();
    const gridSize = new Point(5, 5);
    const snake = new Snake(gridSize);
    let observation = snake.reset();
    SnakeView.Initialize(app.stage, gridSize);
    SnakeView.Draw(observation.collisions, observation.foodPosition);

    const stepMs = 200;
    let accumulatedTime = 0;

    app.ticker.add(async (time) => {
        accumulatedTime += time.deltaMS;

        while (accumulatedTime >= stepMs) {
            accumulatedTime -= stepMs;

            const action = await snakeNet.predict(observation);
            const stepResult = snake.step(action);

            observation = stepResult.observation;
            if (stepResult.terminated)
            {
                observation = snake.reset();
                accumulatedTime = -2000
            }

            SnakeView.Draw(observation.collisions, observation.foodPosition);
        }
    });
})();
