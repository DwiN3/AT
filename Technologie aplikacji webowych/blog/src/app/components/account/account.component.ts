import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AuthService } from '../../services/auth/auth.service';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

@Component({
  selector: 'app-account',
  standalone: true,
  imports: [CommonModule, FormsModule],
  providers: [AuthService],
  templateUrl: './account.component.html',
  styleUrls: ['./account.component.css']
})
export class AccountComponent implements OnInit {
  public username: string | null;
  public email: string | null;
  public role: string | null;
  public newPassword: string = '';
  public MessageText: string = '';
  public isError: boolean = false;

  constructor(
    private authService: AuthService,
    private router: Router) {
    this.username = null;
    this.email = null;
    this.role = null;
  }

  ngOnInit(): void {
    if (this.authService.isLoggedIn()) {
      this.username = this.authService.getUserName();
      this.email = this.authService.getEmail();
      this.role = this.authService.isAdmin() ? 'Administrator' : 'Użytkownik';
    }
  }

  deleteAccount(): void {
    this.authService.deleteAccount().subscribe(
      () => {
        this.MessageText = 'Konto zostało usunięte.';
        this.isError = false;
        this.router.navigate(['/']);
      },
      error => {
        this.MessageText = 'Wystąpił błąd podczas usuwania konta.';
        this.isError = true;
      }
    );
  }

  changePassword(): void {
    this.authService.changePassword(this.newPassword).subscribe(
      (result) => { 
        this.MessageText = 'Hasło zostało zmienione.';
        this.isError = false;
      },
      error => {
        this.MessageText = 'Wystąpił błąd podczas zmiany hasła.';
        this.isError = true;
      }
    );
  }
}
