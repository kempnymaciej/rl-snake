import {Point} from "pixi.js";
import {Array2D} from "./array-2d.ts";

export {Snake, SnakeAction};
export type { SnakeObservation };

enum SnakeAction {
    None = 0,
    TurnLeft = 1,
    TurnRight = 2,
}

interface SnakeObservation {
    headPosition: Point;
    foodPosition: Point;
    direction: Point;
    collisions: Array2D<boolean>;
}

class Snake {
    private _gridSize: Point;

    private _collisions: Array2D<boolean> = new Array2D(0, 0, () => false);
    private _snakeQueue: Point[] = new Array<Point>();
    private _headPosition: Point = new Point(0, 0);
    private _foodPosition: Point = new Point(0, 0);
    private _direction: Point = new Point(0, 0);

    public constructor(gridSize: Point) {
        this._gridSize = gridSize;
        this.reset();
    }

    public reset(): SnakeObservation {
        this._direction = new Point(1, 0);
        this._headPosition = new Point(Math.floor(this._gridSize.x / 2), Math.floor(this._gridSize.y / 2));
        this._snakeQueue = new Array<Point>();
        this._snakeQueue.push(this._headPosition);
        this._collisions = new Array2D(this._gridSize.x, this._gridSize.y, () => false);
        this._collisions.set(this._headPosition.x, this._headPosition.y, true);
        this.positionFood();

        return this.getObservation();
    }

    public step(action: SnakeAction) {
        this.turn(this.actionToFloatDirection(action));

        let terminated = false;
        let nextHeadPosition = new Point(this._headPosition.x + this._direction.x, this._headPosition.y + this._direction.y);

        if (nextHeadPosition.x < 0 || nextHeadPosition.x >= this._gridSize.x || nextHeadPosition.y < 0 || nextHeadPosition.y >= this._gridSize.y) {
            terminated = true;
        }
        else if (this._collisions.get(nextHeadPosition.x, nextHeadPosition.y)) {
            terminated = true;
        }
        else if (nextHeadPosition.equals(this._foodPosition))
        {
            this._headPosition = nextHeadPosition;
            this._snakeQueue.push(nextHeadPosition);
            this._collisions.set(nextHeadPosition.x, nextHeadPosition.y, true);
            this.positionFood()

            if (this._snakeQueue.length === this._gridSize.x * this._gridSize.y)
            {
                terminated = true;
            }
        }
        else {
            const tail = this._snakeQueue.shift();
            if (tail == undefined)
                throw new Error("Snake tail is undefined");
            this._collisions.set(tail.x, tail.y, false);
            this._headPosition = nextHeadPosition;
            this._snakeQueue.push(nextHeadPosition);
            this._collisions.set(nextHeadPosition.x, nextHeadPosition.y, true);
        }

        return {
            observation: this.getObservation(),
            terminated
        };
    }

    private actionToFloatDirection(action: SnakeAction) {
        switch (action) {
            case SnakeAction.TurnLeft:
                return -1;
            case SnakeAction.TurnRight:
                return 1;
            default:
                return 0;
        }
    }

    private turn(direction: number) {
        if (direction == 0)
        {
            return;
        }

        if (this._direction.x == 0){
            this._direction = new Point(direction * this._direction.y, 0)
        }
        else {
            this._direction = new Point(0, direction * -1 * this._direction.x)
        }
    }

    private getObservation() : SnakeObservation{
        return {
            headPosition: this._headPosition,
            foodPosition: this._foodPosition,
            direction: this._direction,
            collisions: this._collisions.clone(),
        }
    }

    private positionFood(){
        const emptyCount = this._gridSize.x * this._gridSize.y - this._snakeQueue.length;

        if (emptyCount === 0)
            return;

        let remaining = Math.floor(Math.random() * emptyCount)

        for (let x = 0; x < this._gridSize.x; x++){
            for (let y = 0; y < this._gridSize.y; y++){
                if (!this._collisions.get(x, y))
                {
                    if (remaining == 0) {
                        this._foodPosition = new Point(x, y);
                        return;
                    }

                    remaining--;
                }
            }
        }
    }
}
