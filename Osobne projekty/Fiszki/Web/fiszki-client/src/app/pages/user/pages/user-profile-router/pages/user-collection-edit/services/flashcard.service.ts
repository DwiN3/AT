import { HttpClient, HttpHeaders } from "@angular/common/http";
import { Injectable } from "@angular/core";
import { map } from "rxjs";
import { FlashcardAddInterface } from "src/app/shared/models/flashcard-add.interface";

@Injectable({providedIn : 'root'})
export class FlashcardService
{
    url : string = 'http://localhost:8080/flashcards/';
    token : string | null = localStorage.getItem('token');
    headers : HttpHeaders = new HttpHeaders({
        'Authorization': `Bearer ${this.token}`,
    });

    constructor(private http : HttpClient){}

    AddFlashcard(flashcard : FlashcardAddInterface)
    {
        return this.http.post<any>(this.url + 'add-flashcard', flashcard, { headers : this.headers })
    }

    DeleteFlashcard(id : number)
    {
        return this.http.delete<any>(this.url + 'delete/' + id, { headers : this.headers});
    }

    GetFlashcard(id : number)
    {
        return this.http.get<any>(this.url + 'show/' + id, { headers : this.headers })
            .pipe(
                map(data => this.ConvertFlashcard(data))
            );
    }

    EditFlashcard(flashcard : FlashcardAddInterface, id : number)
    {
        return this.http.put<any>(`${this.url}edit/${id}`, flashcard, { headers : this.headers });
    }

    private ConvertFlashcard(data : any) : FlashcardAddInterface
    {
        const flashcard : FlashcardAddInterface = 
        {
            id: data.id,
            category: data.category,
            example: data.example,
            translatedExample: data.translatedExample,
            translatedWord: data.translatedWord,
            word: data.word,
            collectionName: data.collectionName
        }

        return flashcard; 
    }

}