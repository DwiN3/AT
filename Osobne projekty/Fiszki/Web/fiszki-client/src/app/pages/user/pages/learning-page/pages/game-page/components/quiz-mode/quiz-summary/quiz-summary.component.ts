import { Component, Input, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { UserLevelModel } from 'src/app/pages/user/models/user-level.model';
import { UserService } from 'src/app/pages/user/services/user.service';

@Component({
  selector: 'app-quiz-summary',
  templateUrl: './quiz-summary.component.html',
  styleUrls: ['./quiz-summary.component.scss']
})
export class QuizSummaryComponent implements OnInit{

  @Input() scoredPoints : number = 0;

  isLoading : boolean = true;
  userLevel : UserLevelModel = {level : 0, nextLevelPoints : 0, points : 0, };
  levelWidth : number = 0;

  constructor(private userService : UserService, private router : Router){}

  ngOnInit(): void 
  {

    this.userService.AddResult(this.scoredPoints)
      .subscribe((data) => {
        this.userLevel.points = 0
        this.userLevel = { level : data.level, nextLevelPoints : data.nextLVLPoints, points : data.points }
        this.LevelWidthCalc();
      }, err => {
        console.log(err);
      }, () => {
        this.isLoading = false;
      })

  }

  LevelWidthCalc() : void
  {
    console.log(this.userLevel);
    const procent = this.userLevel.points / this.userLevel.nextLevelPoints * 100;
    this.levelWidth = procent
    console.log(procent);
  }

  Navigate() : void
  {
    this.router.navigate(['/user'])
  }

}
