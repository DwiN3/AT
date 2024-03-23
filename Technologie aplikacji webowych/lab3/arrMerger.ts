function mergeArrays<T>(arr1: T[], arr2: T[]): T[] {
    return [...arr1, ...arr2]
}

const arr1Num: number[] = [3, 152, 11];
const arr2Num: number[] = [-4, 0, -256];
const mergedArray: number[] = mergeArrays(arr1Num, arr2Num);
console.log("Połączone tablice num: ", mergedArray);

const arr1Str: string[] = ["cukier", "mleko", "płatki"];
const arr2Str: string[] = ["widelec", "łyżka"];
const mergedStringArray: string[] = mergeArrays(arr1Str, arr2Str);
console.log("Połączone tablice str: ", mergedStringArray);