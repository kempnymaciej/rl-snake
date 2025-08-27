export {Array2D};

class Array2D<T> {
    public readonly rowCount: number;
    public readonly columnCount: number;
    private readonly _data: T[];

    constructor(rowCount: number, columnCount: number, initializer : ((row: number, column: number) => T)) {
        this.rowCount = rowCount;
        this.columnCount = columnCount;
        this._data = Array(rowCount * columnCount);

        for (let row = 0; row < rowCount; row++) {
            for (let column = 0; column < columnCount; column++) {
                this._data[row * columnCount + column] = initializer(row, column);
            }
        }
    }

    get(row: number, column: number): T {
        return this._data[row * this.columnCount + column];
    }

    set(row: number, column: number, value: T) {
        this._data[row * this.columnCount + column] = value;
    }

    clone(): Array2D<T> {
        return new Array2D<T>(this.columnCount, this.columnCount, ((x,y) => this.get(x, y)));
    }
}