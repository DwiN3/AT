import React, { useState } from 'react'
import { View, Text, Image, TouchableOpacity, ScrollView, KeyboardAvoidingView, ToastAndroid, Alert, ActivityIndicator, Modal } from 'react-native'
import Ionicons from 'react-native-vector-icons/Ionicons';
import FormInput from '../Components/FormInput';
import { useNavigation } from '@react-navigation/native';
import { styles } from '../Styles/signUpStyles';
import Animated, { FadeInUp, RotateInDownLeft } from 'react-native-reanimated';
import axios from 'axios';
import LoadingModal from '../Components/LoadingModal';

export default function SignUpScreen() {
    const navigation = useNavigation();
    const [personalDataForm, setPersonalDataForm] = useState(true);
    const [loading, setLoading] = useState(false);
    
    const[account, setAccount] = useState({
        email: '',
        password: '',
        passwordRepeted: '',
        contact: {
            name: '',
            lastname: '',
            phoneNumber: '',
            street: '',
            streetNumber: '',
            postalCode: '',
            city: '',
            country: ''
        }
    });

    const [errorMessage, setErrorMessage] = useState ({
        email: '',
        password: '',
        passwordRepeted: '',
        contact: {
            name: '',
            lastname: '',
            phoneNumber: '',
            street: '',
            streetNumber: '',
            postalCode: '',
            city: '',
            country: ''
        }
    });
    
    const validatePersonalData = async () => {
        let errorMessages = {...errorMessage};

        if (account.contact.name.trim() === '') {
            errorMessages.contact.name = 'First name is required!';
        } else {
            errorMessages.contact.name = '';
        }
        if (account.contact.lastname.trim() === '') {
            errorMessages.contact.lastname = 'Last name is required!';
        } else {
            errorMessages.contact.lastname = '';
        }
        if (account.contact.phoneNumber.trim() === '') {
            errorMessages.contact.phoneNumber = 'Phone number is required!';
        } else if (!isValidPhone(account.contact.phoneNumber)) {
            errorMessages.contact.phoneNumber = 'Invalid phone number! Phone number should have 9 digits';
        } else {
            errorMessages.contact.phoneNumber = '';
        }
        if (account.email.trim() === '') {
            errorMessages.email = 'Email address is required!';
        } else if (!isValidEmail(account.email)) {
            errorMessages.email = 'Invalid email address!';
        } else {
            console.log(account.email);
            const email = account.email;
            await axios.post('https://sneakers-api.fly.dev/api/auth/checkEmail', { email })
            .then (response => {
                console.log(response.status === 200);
                errorMessages.email = '';
            }).catch( error => {
                console.log(error.status === 200);
                errorMessages.email = 'Account with this email address already exist!';
            });
        }

        if (account.password.trim() === '') {
            errorMessages.password = 'Password is required!';
        } else if (!isStrongPassword(account.password)) {
            errorMessages.password = 'Password must be at least 8 characters long and include one uppercase letter, one lowercase letter, and one number!';
        } else {
            errorMessages.password = '';
        }
        if (account.passwordRepeted.trim() === '') {
            errorMessages.passwordRepeted = 'Re-password is required!';
        } else if (account.passwordRepeted !== account.password) {
            errorMessages.passwordRepeted = 'Passwords must be the same!';
        } else {
            errorMessages.passwordRepeted = '';
        }

        setErrorMessage(errorMessages);
        return errorMessages;
    };

    const isValidPhone = (phone) => {
        const nineDigitsRegex = /^\d{9}$/;
        const plusElevenDigitsRegex = /^\+\d{11}$/;
    
        return nineDigitsRegex.test(phone);
    };

    const isValidEmail = (email) => {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    };

    const isStrongPassword = (password) => {
        return /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$/.test(password);
    }

    const validateDeliveryData = async () => {
        let errorMessages = {...errorMessage};

        if (account.contact.street.trim() === '') {
            errorMessages.contact.street = 'Street address is required!';
        } else {
            errorMessages.contact.street = '';
        }
        if (account.contact.streetNumber.trim() === '') {
            errorMessages.contact.streetNumber = 'Street number is required!';
        } else {
            errorMessages.contact.streetNumber = '';
        }
        if (account.contact.city.trim() === '') {
            errorMessages.contact.city = 'City is required!';
        } else {
            errorMessages.contact.city = '';
        }
        if (account.contact.postalCode.trim() === '') {
            errorMessages.contact.postalCode = 'Postal code is required!';
        } else if (!isValidPostalCode(account.contact.postalCode)) {
            errorMessages.contact.postalCode = 'Invalid postal code! Postal code should look like 11-222';
        } else {
            errorMessages.contact.postalCode = '';
        }
        if (account.contact.country.trim() === '') {
            errorMessages.contact.country = 'Country is required!';
        } else {
            errorMessages.contact.country = '';
        }

        setErrorMessage(errorMessages);
        return errorMessages;
    }

    const isValidPostalCode = (postalCode) => {
        const postalCodeRegex = /^[0-9]{2}-[0-9]{3}$/;
    
        return postalCodeRegex.test(postalCode);
    };

    const handleCheckPersonalData = async () => {       
        const errorMessages = await validatePersonalData();

        if ( errorMessages.contact.name !== '' || errorMessages.contact.lastname !== '' || errorMessages.contact.phoneNumber !== '' ||
            errorMessages.email !== '' || errorMessages.password !== '' || errorMessages.passwordRepeted !== '' ) {
            ToastAndroid.showWithGravity('Wrong Input Data!', ToastAndroid.SHORT, ToastAndroid.CENTER);
            return;
        }

        setPersonalDataForm(false);
    }

    const handleRegister = async () => {
        const errorMessages = await validateDeliveryData();

        if ( errorMessages.contact.name !== '' || errorMessages.contact.lastname !== '' || errorMessages.contact.phoneNumber !== '' ||
            errorMessages.email !== '' || errorMessages.password !== '' || errorMessages.passwordRepeted !== '' ||
            errorMessage.contact.postalCode !== '' || errorMessage.contact.street !== '' || errorMessage.contact.streetNumber !== ''
            || errorMessage.contact.city !== '' || errorMessage.contact.country !== '' 
        ) {
                ToastAndroid.showWithGravity('Wrong Input Data!', ToastAndroid.SHORT, ToastAndroid.CENTER);
                return;
        }

        register();
    }

    const handleChange = (name, value, nested = false) => {
        /* Phone check */
        if (name === 'phoneNumber' && !/^\d{0,9}$/.test(value)) {
            return;
        }

        /* Postal code check */
        if (name === 'postalCode' && !/^(\d{0,2}-?\d{0,3})?$/.test(value)) {
            return;
        }
        if (nested) {
            setAccount(prevAccount => ({
                ...prevAccount,
                contact: {
                    ...prevAccount.contact,
                    [name]: value
                }
            }));
        } else {
            setAccount(prevAccount => ({
                ...prevAccount,
                [name]: value
            }));
        }
    };

    const showAlert = () => {
        Alert.alert(
            "Konto Utworzone!",
            "Twoje konto zostało pomyślnie utworzone. Prosimy o potwierdzenie rejestracji poprzez kliknięcie w link wysłany na Twój adres e-mail.",
            [
                {
                    text: "OK",
                    onPress: () => console.log("OK Pressed")
                }
            ]
        );
    }

    const handleChangeRoute = () => {
        navigation.navigate('Login');
    };

    const register = () => {
        setLoading(true);
        axios.post('https://sneakers-api.fly.dev/api/auth/register', account).then( response => {
            console.log(response.data);
            setLoading(false);
            ToastAndroid.showWithGravity('Account created succesfully!', ToastAndroid.SHORT, ToastAndroid.CENTER);
            showAlert();
            handleChangeRoute();
        }).catch(error => {
            setLoading(false);
            console.log("Register error!", error);
            
            console.log("Error response data:", error.response.data);
            console.log("Error response status:", error.response.status);
            console.log("Error response headers:", error.response.headers);

            ToastAndroid.showWithGravity('An error occurred. Please try again!', ToastAndroid.SHORT, ToastAndroid.CENTER);
        });
    }

    return (
        <ScrollView style={styles.body}>
            <Image style={styles.background} source={require('../../../../assets/Images/Login/SignUpBackground.png')} />

            <View style={styles.header}>
                <View style={styles.iconBox}>
                <Animated.View style={{transform: [{ rotateZ: '-30deg' }]}}>
                       <Animated.Image entering={RotateInDownLeft.delay(200).duration(1000).springify().damping(3)} style={styles.icon} source={require('../../../../assets/Images/Login/Icon.png')} /> 
                    </Animated.View>
                    <Animated.View style={{transform: [{ rotateZ: '30deg' },{ rotateY: '-180deg' }]}}>
                        <Animated.Image entering={RotateInDownLeft.delay(200).duration(1000).springify().damping(3)} style={styles.icon} source={require('../../../../assets/Images/Login/Icon.png')} />
                    </Animated.View>
                </View>
                <Animated.Image entering={FadeInUp.delay(400).duration(1000).springify().damping(3).randomDelay()} style={[styles.logo]} source={require('../../../../assets/Images/Login/Name.png')} /> 
                <Animated.Text entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.title}>{personalDataForm? 'Register' : 'Shipping details'}</Animated.Text>
            </View>

            <KeyboardAvoidingView behavior='point' style={styles.form}>
                { personalDataForm ? 
                    <>
                        <FormInput iconName='person-outline' iconSize={30} text='Name' value={account.contact.name} handleOnChange={(value) => handleChange('name', value, true)} errorMessage={errorMessage.contact.name}/>
                        <FormInput iconName='person-outline' iconSize={30} text='Surname' value={account.contact.lastname} handleOnChange={(value) => handleChange('lastname', value, true)} errorMessage={errorMessage.contact.lastname} />
                        <FormInput iconName='call-outline' iconSize={30} text='Phone' value={account.contact.phoneNumber} handleOnChange={(value) => handleChange('phoneNumber', value, true)} errorMessage={errorMessage.contact.phoneNumber} />
                        <FormInput iconName='mail-outline' iconSize={30} text='E-mail' value={account.email} handleOnChange={(value) => handleChange('email', value)} errorMessage={errorMessage.email} />
                        <FormInput iconName='lock-closed-outline' iconSize={30} text='Password' value={account.password} secue={true} handleOnChange={(value) => handleChange('password', value)} errorMessage={errorMessage.password} />
                        <FormInput iconName='lock-closed-outline' iconSize={30} text='Re-password' value={account.passwordRepeted} secue={true} handleOnChange={(value) => handleChange('passwordRepeted', value)} errorMessage={errorMessage.passwordRepeted} />

                        <Animated.View entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.buttonBox}>
                            
                            <TouchableOpacity style={styles.button} onPress={() => handleCheckPersonalData()}>
                                <Text style={styles.buttonText}>Continue registration</Text>
                                <Ionicons name='arrow-redo-outline' size={30} color= '#fff' /> 
                            </TouchableOpacity>
                        </Animated.View>

                        <Animated.View entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.signUp}>
                            <Text style={styles.signUpText} >Already have an account? </Text>
                            <TouchableOpacity onPress={() => navigation.push('Login')}>
                                <Text style={styles.signUpLink}>Login</Text>
                            </TouchableOpacity>
                        </Animated.View>
                    </> 
                :
                    <>
                        <>
                            <FormInput iconName='navigate-outline' iconSize={30} text='Street Address' value={account.contact.street} handleOnChange={(value) => handleChange('street', value, true)} errorMessage={errorMessage.contact.street} />
                            <FormInput iconName='locate-outline' iconSize={30} text='Street Number' value={account.contact.streetNumber} handleOnChange={(value) => handleChange('streetNumber', value, true)} errorMessage={errorMessage.contact.streetNumber} />
                            <FormInput iconName='navigate-outline' iconSize={30} text='City' value={account.contact.city} handleOnChange={(value) => handleChange('city', value, true)} errorMessage={errorMessage.contact.city} />
                            <FormInput iconName='locate-outline' iconSize={30} text='Postal / Zpi Code' value={account.contact.postalCode} handleOnChange={(value) => handleChange('postalCode', value, true)} errorMessage={errorMessage.contact.postalCode} />
                            <FormInput iconName='navigate-outline' iconSize={30} text='Country' value={account.contact.country} handleOnChange={(value) => handleChange('country', value, true)} errorMessage={errorMessage.contact.country} />
                        </>
                        <View style={styles.buttonNext2Button}>
                            <Animated.View entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.buttonBoxSmall}>
                                <TouchableOpacity style={styles.button} onPress={() => setPersonalDataForm(!personalDataForm)}>
                                    <Ionicons name='arrow-undo-outline' size={30} color= '#fff' /> 
                                    <Text style={styles.buttonText}>Back</Text>
                                </TouchableOpacity>
                            </Animated.View>

                            <Animated.View entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.buttonBoxSmall}>
                                <TouchableOpacity style={styles.button} onPress={() => handleRegister()}>
                                    <Text style={styles.buttonText}>Register</Text>
                                    <Ionicons name='log-in-outline' size={30} color= '#fff' /> 
                                </TouchableOpacity>
                            </Animated.View>
                        </View>
                   
                        <Animated.View entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.signUp}>
                                <Text style={styles.signUpText} >Already have an account? </Text>
                                <TouchableOpacity onPress={() => navigation.navigate('Login')}>
                                    <Text style={styles.signUpLink}>Login</Text>
                                </TouchableOpacity>
                        </Animated.View>
                    </>
                }
            </KeyboardAvoidingView>
            <LoadingModal visible={loading}/>
        </ScrollView>
    )
}