import { Component, OnInit } from '@angular/core';
import { Router, RouterModule } from "@angular/router";
import { FormsModule } from "@angular/forms";
import { CommonModule } from "@angular/common";
import { AuthService } from "../../services/auth/auth.service";

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [RouterModule, FormsModule, CommonModule],
  providers: [AuthService],
  templateUrl: './navbar.component.html',
  styleUrl: './navbar.component.css'
})
export class NavbarComponent implements OnInit {
  constructor(
    public authService: AuthService, 
    public router: Router) {}

  ngOnInit(): void {}

  logout() {
    this.authService.logout().subscribe((result: any) => {
      this.router.navigate(['/']);
      return result;
    });
  }
}