import {Array2D} from "./array-2d.ts";
import {Container, Graphics, Point} from "pixi.js";

export const SnakeView = {
    Initialize,
    Draw,
}

let nodes: Array2D<Graphics>

function Initialize(container: Container, gridSize: Point){
    const cellSize = 100;

    const background = new Graphics()
        .rect(0, 0, gridSize.x * cellSize, gridSize.y * cellSize)
        .fill(0x000000);
    container.addChild(background);

    nodes = new Array2D<Graphics>(gridSize.x, gridSize.y, (x,y) => {
        const node = new Graphics()
            .rect(x * cellSize, y * cellSize, cellSize, cellSize)
            .fill(0x00ff00);
        container.addChild(node);
        return node;
    });
}

function Draw(snake: Array2D<boolean>, food: Point){

    if (nodes == null)
        return;

    for (let x = 0; x < nodes.rowCount; x++){
        for (let y = 0; y < nodes.columnCount; y++){
            nodes.get(x, y).visible = snake.get(x, y)
        }
    }

    nodes.get(food.x, food.y).visible = true;
}