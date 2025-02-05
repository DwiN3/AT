"use client"

import React, { FormEvent, useState } from "react";
import { Button } from "@nextui-org/button";
import { Input } from "@nextui-org/input";
import { Eye, EyeOff, LockKeyhole } from "lucide-react";
import toast from "react-hot-toast";

import { validatePassword } from "@/lib/formValidators";
import { useAuth } from "@/providers/authProvider";


const CHANGE_PASSWORD_URL = "api/auth/change-password"

export default function ChangePassword() {
    const [isOldPasswordVisible, setIsOldPasswordVisible] = React.useState(false);
    const [isNewPasswordVisible, setIsNewPasswordVisible] = React.useState(false);
    const [isConfirmPasswordVisible, setIsConfirmPasswordVisible] = React.useState(false);

    const [passwordChangeError, setPasswordChangeError] = useState(0);
    const [passwordChangeMessage, setPasswordChangeMessage] = useState("Podaj obecne hasło oraz nowe, aby dokonać zmiany.");
    const [isOldPasswordInvalid, setIsOldPasswordInvalid] = useState(false);
    const [isNewPasswordsInvalid, setIsNewPasswordsInvalid] = useState(false);
    const [formData, setFormData] = useState({
        old_password: "",
        new_password: "",
        confirm_password: ""
    });
    const auth = useAuth();

    const toggleVisibilityOldPassword = () => setIsOldPasswordVisible(!isOldPasswordVisible);
    const toggleVisibilityNewPassword = () => setIsNewPasswordVisible(!isNewPasswordVisible);
    const toggleVisibilityConfirmPassword = () => setIsConfirmPasswordVisible(!isConfirmPasswordVisible);

    const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setIsOldPasswordInvalid(false);
        setIsNewPasswordsInvalid(false);

        if (isInvalidPassword || formData.old_password == "" || formData.new_password == "" || formData.confirm_password == "") {
            setPasswordChangeError(1);
            setPasswordChangeMessage("Aby zmienić hasło musisz wprowadzić poprawnie wszystkie dane!");
        } else if (isInvalidConfirmPassword) {
            setPasswordChangeError(1);
            setPasswordChangeMessage("Podane hasła różnią się od siebie!");
        } else {
            const requestOptions = {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(formData)
            };

            const response = await fetch(CHANGE_PASSWORD_URL, requestOptions);

            interface RegisterResponse {
                message?: string;
            }
            let data: RegisterResponse = {};

            try {
                data = await response.json();
            } catch { }

            if (response.status == 200) {
                setPasswordChangeError(2);
                setPasswordChangeMessage("Hasło zostało zmienione.");
                showToast(false);
            } else if (response.status == 400) {
                setPasswordChangeError(1);
                const errorResponse = data?.message;

                if (errorResponse == "Old password incorrect.") {
                    setIsOldPasswordInvalid(true);
                    setPasswordChangeMessage("Obecne hasło, które podałeś jest niepoprawne.")
                } else if (errorResponse == "New passwords do not match.") {
                    setIsNewPasswordsInvalid(true);
                    setPasswordChangeMessage("Nowe hasła nie pasują do siebie.")
                } else {
                    setPasswordChangeMessage("Podczas zmiany hasła wystąpił nieoczekiwany błąd. Spróbuj ponownie później.");
                }

                showToast(true);
            } else if (response.status == 401) {
                auth.loginRequired();
                showToast(true);
            } else {
                setPasswordChangeError(1)
                setPasswordChangeMessage("Podczas zmiany hasła wystąpił nieoczekiwany błąd serwera. Spróbuj ponownie później.");
                showToast(true);
            }
        }
    }

    const showToast = async (isError: boolean) => {
        toast(isError ? 'Wystąpił błąd podczas zmiany hasła!' : 'Aktualizacja hasła zakończona sukcesem!',
            {
                icon: isError ? '☹️' : '🔐',
                style: {
                    borderRadius: '16px',
                    textAlign: "center",
                    padding: '16px',
                    background: isError ? "#F31260" : "#006FEE",
                    color: '#fff',
                },
            }
        );
    }

    const isInvalidPassword = React.useMemo(() => {
        if (formData.new_password === "") return false;

        return validatePassword(formData.new_password) ? false : true;
    }, [formData.new_password]);

    const isInvalidConfirmPassword = React.useMemo(() => {
        if (formData.confirm_password === "") return false;

        return formData.new_password == formData.confirm_password ? false : true;
    }, [formData.confirm_password]);

    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setIsOldPasswordInvalid(false);
        setIsNewPasswordsInvalid(false);

        setFormData({
            ...formData,
            [event.target.name]: event.target.value
        });
    }

    return (
        <div className="py-4">
            <h1 className="text-primary text-2xl font-semibold mb-2">Zmiana hasła</h1>
            <p className={`${passwordChangeError == 1 ? "text-danger-500" : passwordChangeError == 2 ? "text-success-500" : "text-default-600"}`}>{passwordChangeMessage}</p>
            <form className="overflow-visible flex flex-col gap-3 mt-6" onSubmit={handleSubmit}>
                <Input
                    autoComplete="username"
                    className="invisible h-0"
                    color="default"
                />
                <Input
                    autoComplete="current-password"
                    color="default"
                    endContent={
                        <button aria-label="toggle password visibility" className="focus:outline-none" type="button" onClick={toggleVisibilityOldPassword}>
                            {isOldPasswordVisible ? (
                                <EyeOff className={`text-2xl pointer-events-none ${isOldPasswordInvalid ? "text-danger-400" : "text-default-400"}`} />
                            ) : (
                                <Eye className={`text-2xl pointer-events-none ${isOldPasswordInvalid ? "text-danger-400" : "text-default-400"}`} />
                            )}
                        </button>
                    }
                    errorMessage="Podane hasło jest niepoprawne."
                    isInvalid={isOldPasswordInvalid}
                    isRequired={true}
                    label="Obecne Hasło"
                    labelPlacement="outside"
                    name="old_password"
                    placeholder="Hasło"
                    size="md"
                    startContent={
                        <LockKeyhole className={`text-2xl  pointer-events-none flex-shrink-0 ${isOldPasswordInvalid ? "text-danger-400" : "text-default-400"}`} />
                    }
                    type={isOldPasswordVisible ? "text" : "password"}
                    value={formData.old_password}
                    onChange={handleChange}
                />
                <Input
                    autoComplete="new-password"
                    color="default"
                    endContent={
                        <button aria-label="toggle password visibility" className="focus:outline-none" type="button" onClick={toggleVisibilityNewPassword}>
                            {isNewPasswordVisible ? (
                                <EyeOff className={`text-2xl pointer-events-none ${isInvalidPassword || isOldPasswordInvalid ? "text-danger-400" : "text-default-400"}`} />
                            ) : (
                                <Eye className={`text-2xl pointer-events-none ${isInvalidPassword || isOldPasswordInvalid ? "text-danger-400" : "text-default-400"}`} />
                            )}
                        </button>
                    }
                    errorMessage="Hasło musi posiadać co najmniej 8 znaków, w tym 1 małą literę, 1 dużą literę, cyfrę oraz znak specjalny."
                    isInvalid={isInvalidPassword || isNewPasswordsInvalid}
                    isRequired={true}
                    label="Nowe Hasło"
                    labelPlacement="outside"
                    name="new_password"
                    placeholder="Nowe hasło"
                    size="md"
                    startContent={
                        <LockKeyhole className={`text-2xl  pointer-events-none flex-shrink-0 ${isInvalidPassword || isNewPasswordsInvalid ? "text-danger-400" : "text-default-400"}`} />
                    }
                    type={isNewPasswordVisible ? "text" : "password"}
                    value={formData.new_password}
                    onChange={handleChange}
                />
                <Input
                    autoComplete="re-new-password"
                    color="default"
                    endContent={
                        <button aria-label="toggle password visibility" className="focus:outline-none" type="button" onClick={toggleVisibilityConfirmPassword}>
                            {isConfirmPasswordVisible ? (
                                <EyeOff className={`text-2xl pointer-events-none ${isInvalidConfirmPassword ? "text-danger-400" : "text-default-400"}`} />
                            ) : (
                                <Eye className={`text-2xl pointer-events-none ${isInvalidConfirmPassword ? "text-danger-400" : "text-default-400"}`} />
                            )}
                        </button>
                    }
                    errorMessage="Hasła nie mogą się od siebie różnić!"
                    isInvalid={isInvalidConfirmPassword}
                    isRequired={true}
                    label="Potwierdź Nowe Hasło"
                    labelPlacement="outside"
                    name="confirm_password"
                    placeholder="Potwierdź nowe hasło"
                    size="md"
                    startContent={
                        <LockKeyhole className={`text-2xl  pointer-events-none flex-shrink-0 ${isInvalidConfirmPassword ? "text-danger-400" : "text-default-400"}`} />
                    }
                    type={isConfirmPasswordVisible ? "text" : "password"}
                    value={formData.confirm_password}
                    onChange={handleChange}
                />
                <Button className="w-full mt-4" color="primary" size="sm" type="submit" variant="shadow">
                    Zmień hasło
                </Button>
            </form>
        </div>
    );
}