import { Component, OnDestroy, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { Store } from '@ngrx/store';
import { Observable, Subscription } from 'rxjs';
import { BaseFlashcardInterface } from 'src/app/shared/models/flashcard.interface';
import { GameSettingsState } from '../../../../store/game.state';

@Component({
  selector: 'app-learning-mode',
  templateUrl: './learning-mode.component.html',
  styleUrls: ['./learning-mode.component.scss']
})
export class LearningModeComponent implements OnInit, OnDestroy{

  gameSettings$! : Observable<GameSettingsState>
  private gameSettingsSubscription: Subscription | undefined;

  flashcards : BaseFlashcardInterface[] = [];
  translatedWord : boolean = false;
  changeFlashcard : boolean = false;
  round : number = 0;
  summary : boolean = false;

  constructor(private store : Store<{gameSettings : GameSettingsState}>, private router : Router){}

  ngOnInit(): void 
  {
    this.gameSettings$ = this.store.select('gameSettings');

    this.gameSettingsSubscription = this.gameSettings$
      .subscribe(data => {
        this.flashcards = data.flashcards;
        this.translatedWord = data.polishFirst;
      })

      if(this.flashcards.length === 0)
        this.router.navigate(['/user/learning']);
  }

  ngOnDestroy(): void 
  {
    if (this.gameSettingsSubscription) {
      this.gameSettingsSubscription.unsubscribe();
    }
  }

  RevealFlashcard() : void
  {

    if(this.round + 1 === this.flashcards.length && this.changeFlashcard === true)
      this.summary = true;

    this.translatedWord = !this.translatedWord;
    
    if(this.changeFlashcard)
    {
      this.changeFlashcard = false;
      this.round++;
      return;
    }

    this.changeFlashcard = true;
  }

  PlayAgain() : void
  {
    this.round = 0;
    this.summary = false;
  }

}

  
