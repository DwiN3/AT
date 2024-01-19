import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UserProfileRouterComponent } from './user-profile-router.component';

describe('UserProfileRouterComponent', () => {
  let component: UserProfileRouterComponent;
  let fixture: ComponentFixture<UserProfileRouterComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [UserProfileRouterComponent]
    });
    fixture = TestBed.createComponent(UserProfileRouterComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
