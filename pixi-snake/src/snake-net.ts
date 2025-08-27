import * as ort from "onnxruntime-web";
import {SnakeAction, SnakeObservation} from "./snake.ts";

const MODEL_PATH = "/assets/model.onnx";

export {SnakeNet};

class SnakeNet {

    private _session: ort.InferenceSession;

    public static async create() {
        const session = await ort.InferenceSession.create(MODEL_PATH);
        return new SnakeNet(session);
    }

    public constructor(session: ort.InferenceSession) {
        this._session = session;
    }

    public async predict(input: SnakeObservation) {
        const processedObservation = new Float32Array(input.collisions.rowCount * input.collisions.columnCount + 2 + 2 + 2);

        for (let x = 0; x < input.collisions.rowCount; x++){
            for (let y = 0; y < input.collisions.columnCount; y++){
                processedObservation[x * input.collisions.columnCount + y] = input.collisions.get(x, y) ? 1 : 0;
            }
        }

        const collisionsLength = input.collisions.rowCount * input.collisions.columnCount;
        processedObservation[collisionsLength] = input.direction.x;
        processedObservation[collisionsLength + 1] = input.direction.y;
        processedObservation[collisionsLength + 2] = input.foodPosition.x;
        processedObservation[collisionsLength + 3] = input.foodPosition.y;
        processedObservation[collisionsLength + 4] = input.headPosition.x;
        processedObservation[collisionsLength + 5] = input.headPosition.y;

        const tensor = new ort.Tensor("float32", processedObservation, [1, processedObservation.length]);

        const result = await this._session.run({input: tensor});

        const output = result.output.data; // Float32Array, e.g. [0.1, 0.05, 0.8, 0.05]

        let maxIndex = 0;
        let maxValue = output[0];
        for (let i = 1; i < output.length; i++) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                maxIndex = i;
            }
        }
        return maxIndex as SnakeAction;
    }
}