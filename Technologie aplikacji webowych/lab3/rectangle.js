"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Rectangle = exports.Point = void 0;
var Point = /** @class */ (function () {
    function Point(x, y) {
        this.x = x;
        this.y = y;
    }
    Point.prototype.move = function (x, y) {
        this.x += x;
        this.y += y;
    };
    return Point;
}());
exports.Point = Point;
var Rectangle = /** @class */ (function () {
    function Rectangle(topLeft, topRight, bottomLeft, bottomRight) {
        this.topLeft = topLeft;
        this.topRight = topRight;
        this.bottomLeft = bottomLeft;
        this.bottomRight = bottomRight;
    }
    Rectangle.prototype.move = function (x, y) {
        this.topLeft.move(x, y);
        this.topRight.move(x, y);
        this.bottomLeft.move(x, y);
        this.bottomRight.move(x, y);
    };
    Rectangle.prototype.getArea = function () {
        var width = Math.abs(this.topRight.x - this.topLeft.x);
        var height = Math.abs(this.topLeft.y - this.bottomLeft.y);
        return width * height;
    };
    return Rectangle;
}());
exports.Rectangle = Rectangle;
var topLeft = new Point(1, 1);
var topRight = new Point(5, 1);
var bottomLeft = new Point(1, 4);
var bottomRight = new Point(5, 4);
var rectangle = new Rectangle(topLeft, topRight, bottomLeft, bottomRight);
console.log('\nPole = ' + rectangle.getArea());
console.log('Punkty:                 ', rectangle.topLeft);
rectangle.move(2, 2);
console.log('Punkty po przesunięciu: ', rectangle.topLeft);
