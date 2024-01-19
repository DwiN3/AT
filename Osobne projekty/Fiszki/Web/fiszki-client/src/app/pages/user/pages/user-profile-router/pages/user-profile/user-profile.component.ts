import { Component, ComponentFactoryResolver, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { UserLevelModel } from '../../../../models/user-level.model';
import { UserService } from '../../../../services/user.service';

@Component({
  selector: 'app-user-profile',
  templateUrl: './user-profile.component.html',
  styleUrls: ['./user-profile.component.scss']
})
export class UserProfileComponent implements OnInit{

  levelInfo : UserLevelModel | null = null;
  isLoading : boolean = true;

  constructor(private route : ActivatedRoute, private componentFactoryResolver : ComponentFactoryResolver){}

  ngOnInit(): void {
      this.route.data
        .subscribe(res => {
          const data = (res['levelInfo']);
          this.levelInfo = new UserLevelModel(data.level, data.nextLVLPoints, data.points)
          this.isLoading = false;
        }, err => {
          this.isLoading = false;
        })
  }

}
