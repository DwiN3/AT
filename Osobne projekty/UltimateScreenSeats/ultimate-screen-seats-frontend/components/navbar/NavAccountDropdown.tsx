import { Avatar } from "@nextui-org/avatar";
import { Dropdown, DropdownItem, DropdownMenu, DropdownTrigger } from "@nextui-org/dropdown";

import { useAuth } from "@/providers/authProvider";


export default function NavbarAccount() {
  const auth = useAuth();

  const handleLogout = () => {
    auth.logout();
  }

  return (
    <div className="flex items-center gap-4">
      <Dropdown placement="bottom-end">
        <DropdownTrigger>
          <Avatar
            isBordered
            as="button"
            className="transition-transform"
            src="https://i.pravatar.cc/150?u=a042581f4e29026024d"
          />
        </DropdownTrigger>
        <DropdownMenu aria-label="Profile Actions" variant="flat">
          <DropdownItem key="profile" className="h-14 gap-2">
            <p className="font-semibold">Zalogowano jako</p>
            <p className="font-semibold">{auth.username}</p>
          </DropdownItem>
          <DropdownItem key="reservations" href="/reservations">
            Rezerwacje
          </DropdownItem>
          <DropdownItem key="settings" href="/profile">
            Ustawienia Konta
          </DropdownItem>
          <DropdownItem key="logout" color="danger" onClick={handleLogout}>
            Wyloguj się
          </DropdownItem>
        </DropdownMenu>
      </Dropdown>
    </div>
  );
}