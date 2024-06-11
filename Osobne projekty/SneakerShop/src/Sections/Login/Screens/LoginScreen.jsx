import React, { useState } from 'react'
import { View, Text, Image, TouchableOpacity, KeyboardAvoidingView, ScrollView, ToastAndroid, Alert } from 'react-native'
import FormInput from '../Components/FormInput';
import { styles } from '../Styles/loginStyles'
import { useNavigation } from '@react-navigation/native';
import Animated, { FadeInUp, RotateInDownLeft } from 'react-native-reanimated';
import axios from 'axios';
import * as Keychain from 'react-native-keychain';
import LoadingModal from '../Components/LoadingModal';

export default function LoginScreen() {
    const navigation = useNavigation();
    const [modalVisible, setModalVisibility] = useState(false);

    const [login, setLogin] = useState({
        email: '',
        password: ''
    });

    const [errorMessage, setErrorMessage] = useState ({
        email: '',
        password: ''
    });

    const validation = async () => {
        let errorMessages = {...errorMessage};

        if (login.email.trim() === '') {
            errorMessages.email = 'Email address is required!';
        } else if (!isEmailAvailable(login.email)) {
            errorMessages.email = 'Account with given email address doesn\'t exist!';
        } else {
            errorMessages.email = '';
        }
        if (login.password.trim() === '') {
            errorMessages.password = 'Password is required!';
        } else {
            errorMessages.password = '';
        }

        setErrorMessage(errorMessages);
        return errorMessages;
    }

    const isEmailAvailable = (email) => {
        return true;
    }

    const handleLogin = async () => {
        const errorMessages = await validation();

        if ( errorMessages.email !== '' || errorMessages.password !== '' ) {
            ToastAndroid.showWithGravity('Wrong Input Data!', ToastAndroid.SHORT, ToastAndroid.CENTER);
            return;
        }

        logIn();
    }

    const handleChange = (name, value) => {
        setLogin(prevLogin => ({
            ...prevLogin,
            [name]: value
        }));
    };

    const showAlertEmailNotConfirmed = async (email) => {
        const sendConfirmEmail = async () => {
            try {
                const response = await axios.post('https://sneakers-api.fly.dev/api/auth/sendConfirmEmail', { email });
                console.log('Confirmation email sent:', response.data);
            } catch (error) {
                console.error('Failed to send confirmation email', error);
            }
        };
    
        Alert.alert(
            "Email niepotwierdzony!",
            "Aby zalogować się na to konto musisz potwierdzić swoje konto. \nMożesz to zrobić poprzez kliknięcie w link na swoim koncie emial.",
            [
                {
                    text: "OK",
                    onPress: () => console.log("OK Pressed")
                },
                {
                    text: "Wyślij wiadomość ponownie",
                    onPress: sendConfirmEmail
                }
            ]
        );
    };

    const logIn = () => {
        setModalVisibility(true);
        axios.post('https://sneakers-api.fly.dev/api/auth/login', login).then(response => {
            const userData = {
                email: response.data.email,
                emailConfirmed: response.data.emailConfirmed,
                roles: response.data.roles,
                token: response.data.token
            };

            Keychain.setGenericPassword('USER-DATA', JSON.stringify(userData))
                .then(() => {
                    console.log('User data saved successfully');
                })
                .catch(error => {
                    console.error('Failed to save user data', error);
            });

            if (response.data.roles[0].name === "User") {
                navigation.navigate("MainMenu");
            } else if (response.data.roles[0].name === "Employee") {
                navigation.navigate("AddProduct");
            } else {
                navigation.navigate("Login");
                ToastAndroid.showWithGravity('An unexpected error occurred while logging in. Please try again later.', ToastAndroid.SHORT, ToastAndroid.CENTER);
                setModalVisibility(false);
                return;
            }

            if (!response.data.emailConfirmed) {
                showAlertEmailNotConfirmed(response.data.email);
            }
            setModalVisibility(false);
        }).catch(error => {
            setModalVisibility(false);
            console.log("Login error!", error);
            ToastAndroid.showWithGravity('No account found. Please check your email and password.', ToastAndroid.SHORT, ToastAndroid.CENTER);
            
            let errorMessages = {...errorMessage};
            errorMessages.email = 'Invalid Email or Password!';
            errorMessages.password = 'Invalid Email or Password!';
            setErrorMessage(errorMessages);
        });
    }


    return (
        <ScrollView style={styles.body}>
            <Image style={styles.background} source={require('../../../../assets/Images/Login/LoginBack.png')} />

            <View style={styles.header}>
            <Animated.Image entering={FadeInUp.delay(400).duration(1000).springify().damping(3)} style={[styles.logo]} source={require('../../../../assets/Images/Login/Name.png')} />    
                <View style={styles.iconBox}>
                    <Animated.View style={{transform: [{ rotateZ: '-30deg' }]}}>
                       <Animated.Image entering={RotateInDownLeft.delay(200).duration(1000).springify().damping(3)} style={styles.icon} source={require('../../../../assets/Images/Login/Icon.png')} /> 
                    </Animated.View>
                    <Animated.View style={{transform: [{ rotateZ: '30deg' },{ rotateY: '-180deg' }]}}>
                        <Animated.Image entering={RotateInDownLeft.delay(200).duration(1000).springify().damping(3)} style={styles.icon} source={require('../../../../assets/Images/Login/Icon.png')} />
                    </Animated.View>
                </View>
                <Animated.Text entering={FadeInUp.duration(1000).springify()} style={styles.title}>Login</Animated.Text>
            </View>

            <KeyboardAvoidingView behavior='height' keyboardVerticalOffset={-100} style={styles.form}>
                <FormInput iconName='mail-outline' iconSize={30} text='E-mail' value={login.email} handleOnChange={(value) => handleChange('email', value)} errorMessage={errorMessage.email} />
                <FormInput iconName='lock-closed-outline' iconSize={30} text='Password' value={login.password} secue={true} handleOnChange={(value) => handleChange('password', value)} errorMessage={errorMessage.password} />
                
                <Animated.View entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.buttonBox}>
                    <TouchableOpacity onPress={() => handleLogin()} style={styles.button}>
                        <Text style={styles.buttonText}>Login</Text>
                    </TouchableOpacity>
                </Animated.View>
                
                <Animated.View entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.signUp}>
                    <Text style={styles.signUpText} >Don't have an account? </Text>
                    <TouchableOpacity onPress={() => navigation.navigate('SignUp')}>
                        <Text style={styles.signUpLink}>SignUp</Text>
                    </TouchableOpacity>
                </Animated.View>
            
                {/* Developer login  */}
                {/* <View style={{ display: "flex", alignItems: "center", backgroundColor: "#411c5d", width: 300, borderRadius: 20 }}>
                    <TouchableOpacity onPress={() => setLogin({ email: "michalpl6188@gmail.com", password: "Qwerty1234%" })}>
                        <Text style={[styles.buttonText, {textAlign: 'center'}]}>Worker</Text>
                    </TouchableOpacity>
                    <TouchableOpacity onPress={() => setLogin({ email: "michalpl918@gmail.com", password: "Qwerty1234%" })}>
                        <Text style={[styles.buttonText, {textAlign: 'center'}]}>User</Text>
                    </TouchableOpacity>
                </View> */}
                </KeyboardAvoidingView>
                <LoadingModal visible={modalVisible}/>
        </ScrollView>
    )
}