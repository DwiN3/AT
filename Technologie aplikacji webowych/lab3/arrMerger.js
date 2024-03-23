var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
function mergeArrays(arr1, arr2) {
    return __spreadArray(__spreadArray([], arr1, true), arr2, true);
}
var arr1Num = [3, 152, 11];
var arr2Num = [-4, 0, -256];
var mergedArray = mergeArrays(arr1Num, arr2Num);
console.log("Połączone tablice num: ", mergedArray);
var arr1Str = ["cukier", "mleko", "płatki"];
var arr2Str = ["widelec", "łyżka"];
var mergedStringArray = mergeArrays(arr1Str, arr2Str);
console.log("Połączone tablice str: ", mergedStringArray);
