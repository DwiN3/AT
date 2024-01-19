import { Injectable } from "@angular/core";
import { BaseFlashcardInterface } from "src/app/shared/models/flashcard.interface";
import { QuizItemInterface } from "../models/quiz-item.model";

@Injectable({providedIn : 'root'})
export class CreateQuizService
{
    CreateQuiz(flashcards : BaseFlashcardInterface[], polishFirst : boolean) : QuizItemInterface[]
    {
    const quizItems : QuizItemInterface[] = [];

    flashcards.forEach(flashcard => {
      
      const question = polishFirst === true ? flashcard.word : flashcard.translatedWord;
      const correctAnswer = polishFirst === true ? flashcard.translatedWord : flashcard.word;
      const wariants = this.CreateWariants(flashcards, correctAnswer, polishFirst);

      const item : QuizItemInterface = {
        question : question,
        correctAnswer : correctAnswer,
        wariants : this.ShuffleArray(wariants)
      }

      quizItems.push(item);
    })
        return quizItems;
    }

    private CreateWariants(flashcards : BaseFlashcardInterface[], correctAnswer : string, polishFirst : boolean) : string[]
  {
    const wariants : string [] = [correctAnswer];
    for(let i = 0; i < 3; i++)
    {
      let randomIndex;
      let randomWord;
      if(polishFirst === true)
      {
          randomIndex = Math.floor(Math.random() * flashcards.length);
          randomWord = flashcards[randomIndex].translatedWord;

          do
          {
              randomIndex = Math.floor(Math.random() * flashcards.length);
              randomWord = flashcards[randomIndex].translatedWord;
          } while(correctAnswer.includes(randomWord) || wariants.includes    (randomWord))
      }
      else
      {
          randomIndex = Math.floor(Math.random() * flashcards.length);
          randomWord = flashcards[randomIndex].word;

          do
          {
              randomIndex = Math.floor(Math.random() * flashcards.length);
              randomWord = flashcards[randomIndex].word;
          } while(correctAnswer.includes(randomWord) || wariants.includes(randomWord))  
      }
      wariants.push(randomWord);
    }
    return wariants;
  }

  private ShuffleArray(wariants : string[]) : string[]
  {
    for(let i = wariants.length - 1; i > 0; i--)
    {
      let j = Math.floor(Math.random() * (i + 1));
      [wariants[i], wariants[j]] = [wariants[j], wariants[i]];
    }
    return wariants;
  }

}