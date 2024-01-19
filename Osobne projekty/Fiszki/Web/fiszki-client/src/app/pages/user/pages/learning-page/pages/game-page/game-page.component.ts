import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { Store } from '@ngrx/store';
import { Observable, Subscription } from 'rxjs';
import { GameSettingsState } from '../../store/game.state';

@Component({
  selector: 'app-game-page',
  templateUrl: './game-page.component.html',
  styleUrls: ['./game-page.component.scss']
})
export class GamePageComponent implements OnInit{
  
  constructor(private store : Store<{gameSettings : GameSettingsState}>){}

  gameSettings$! : Observable<GameSettingsState>
  private gameSettingsSubscription: Subscription | undefined;

  learningMode : boolean | null = null;
  polishFirst : boolean | null = null;

  ngOnInit(): void 
  {
    this.gameSettings$ = this.store.select('gameSettings');
    this.gameSettingsSubscription = this.gameSettings$
      .subscribe(data => {
        if(data.learningMode === 'learning')
          this.learningMode = true;
        else 
          this.learningMode = false;
          
        this.polishFirst = data.polishFirst;
      })
  }

  ngOnDestroy(): void 
  {
    if (this.gameSettingsSubscription) {
      this.gameSettingsSubscription.unsubscribe();
    }
  }


}
