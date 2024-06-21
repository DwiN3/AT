import { Component, OnInit } from '@angular/core';
import { RouterModule } from "@angular/router";
import { CommonModule } from "@angular/common";
import { AuthService } from '../../services/auth/auth.service';

@Component({
  selector: 'home',
  standalone: true,
  imports: [RouterModule, CommonModule],
  providers: [AuthService],
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit {
  public userName: string | null;

  constructor(public authService: AuthService) {
    this.userName = null;
  }

  ngOnInit(): void {
    if (this.authService.isLoggedIn()) {
      this.userName = this.authService.getUserName();
    }
  }
}
