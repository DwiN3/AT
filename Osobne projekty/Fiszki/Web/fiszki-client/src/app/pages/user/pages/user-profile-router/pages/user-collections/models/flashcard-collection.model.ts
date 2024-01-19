export class FlashcardCollectionModel
{
    collectionName : string;
    flashcardsQuantity : number;

    constructor(collectionName : string, flashcardsQuantity : number)
    {
        this.collectionName = collectionName;
        this.flashcardsQuantity = flashcardsQuantity;
    }
}