"use client"

import React, { FormEvent, useState } from "react";
import { Button } from "@nextui-org/button";
import { Card, CardBody, CardFooter, CardHeader } from "@nextui-org/card";
import { Input } from "@nextui-org/input";
import { Link } from "@nextui-org/link";
import { Eye, EyeOff, LockKeyhole, Mail } from "lucide-react";
import toast from "react-hot-toast";

import { useAuth } from "@/providers/authProvider";
import { validateEmail } from "@/lib/formValidators";


const LOGIN_URL = "api/auth/login"

export default function Page() {
    const [isVisible, setIsVisible] = React.useState(false);
    const [loginMessage, setLoginMessage] = useState("Wprowadź adres email oraz hasło, aby uzyskać dostęp do swojego konta.");
    const [loginError, setLoginError] = useState(false);
    const [formData, setFormData] = useState({
        email: "",
        password: ""
    });

    const auth = useAuth();
    const toggleVisibility = () => setIsVisible(!isVisible);

    const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (formData.email == "" || formData.password == "") {
            setLoginError(true);
            setLoginMessage("Aby się zalogować musisz podać poprawny adres email oraz hasło.");
            showToast(true);
        } else if (isInvalidEmail) {
            setLoginError(true);
            setLoginMessage("Podany adres email ma niepoprawny format.");
            showToast(true);
        } else {
            const requestOptions = {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(formData)
            };

            const response = await fetch(LOGIN_URL, requestOptions);

            interface LoginResponse {
                username?: string,
                access?: string,
                refresh?: string,
                detail?: string,
                role?: string
            };
            let data: LoginResponse = {};

            try {
                data = await response.json();
            } catch { }

            if (response.status == 200) {
                auth.login(data.username, data.role, data.access, data.refresh);
                setLoginError(false);
                showToast(false);
            } else if (response.status == 400 || response.status == 401) {
                setLoginError(true);
                setLoginMessage("Podany adres email lub hasło są niepoprawne. Sprawdź poprawność wprowadzoanych danych i spróbuj ponownie.");
                showToast(true);
            } else {
                setLoginError(true);
                setLoginMessage("Podczas logowania wystąpił nieoczekiwany błąd servera. Spróbuj ponownie później.");
                showToast(true);
            }
        }
    }

    const showToast = async (isError: boolean) => {
        toast(isError? 'Podczas logowania wystąpił błąd. Spróbuj ponownie później!': 'Pomyślnie zalogowano. Witamy!',
            {
                icon: isError? '☹️' : '👏',
                style: {
                borderRadius: '16px',
                textAlign: "center",
                padding: '16px',
                background: isError? "#F31260" : "#006FEE",
                color: '#fff',
                },
            }
        );
    }

    const isInvalidEmail = React.useMemo(() => {
        if (formData.email === "") return false;
    
        return validateEmail(formData.email) ? false : true;
      }, [formData.email]);

    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({
            ...formData,
            [event.target.name]: event.target.value
        });

        if (loginError) {
            setLoginError(false);
            setLoginMessage("Wprowadź poprawny adres email oraz hasło, aby uzyskać dostęp do swojego konta.");
        }
    }
    
  return (
    <div className="flex justify-center items-center h-full">
        <form onSubmit={handleSubmit}>
            <Card className="sm:w-[32rem] w-full p-4">
                <CardHeader className="pb-0 pt-2 px-4 flex-col items-start">
                    <h1 className="text-primary text-4xl font-semibold mb-2">Logowanie</h1>
                    <p className={`${loginError? "text-danger-500": "text-default-600"}`}>{loginMessage}</p>
                </CardHeader>
                <CardBody className="overflow-visible flex flex-col gap-4 mt-2">
                <Input
                    color={loginError? "danger" : "default"}
                    errorMessage="Sprawdź podany adres email!"
                    isInvalid={loginError}
                    isRequired={true}
                    label="Email"
                    labelPlacement="outside"
                    name="email"
                    placeholder="Email"
                    size="lg"
                    startContent={
                        <Mail className={`text-2xl  pointer-events-none flex-shrink-0 ${loginError? "text-danger-400" : "text-default-400"}`}/>
                    }
                    type="email"
                    value={formData.email}
                    onChange={handleChange}
                    />
                <Input
                    color={loginError? "danger" : "default"}
                    endContent={
                        <button aria-label="toggle password visibility" className="focus:outline-none" type="button" onClick={toggleVisibility}>
                        {isVisible ? (
                            <EyeOff className={`text-2xl pointer-events-none ${loginError? "text-danger-400" :"text-default-400"}`} />
                        ) : (
                            <Eye className={`text-2xl pointer-events-none ${loginError? "text-danger-400" :"text-default-400"}`} />
                        )}
                        </button>
                    }
                    errorMessage="Sprawdź podane hasło!"
                    isInvalid={loginError}
                    isRequired={true}
                    label="Hasło"
                    labelPlacement="outside"
                    name="password"
                    placeholder="Hasło"
                    size="lg"
                    startContent={
                        <LockKeyhole className={`text-2xl  pointer-events-none flex-shrink-0 ${loginError? "text-danger-400" :"text-default-400"}`}/>
                    }
                    type={isVisible ? "text" : "password"}
                    value={formData.password}
                    onChange={handleChange}
                    />
                </CardBody>
                <CardFooter className="flex flex-col">
                    <Link className="self-end mb-2" color="primary" href="#" size="md" underline="hover">
                        Zapomniałeś hasła?
                    </Link>
                    <Button className="w-full" color="primary" size="md" type="submit" variant="shadow">
                        Zaloguj się
                    </Button> 
                    <div className="flex gap-2 mt-4">
                        <p>Lub jeśli nie masz konta</p>
                        <Link color="primary" href="/register" size="md" underline="hover">
                            Zarejestruj się
                        </Link>
                    </div>
                </CardFooter>
            </Card>
        </form>
    </div>
  );
}
