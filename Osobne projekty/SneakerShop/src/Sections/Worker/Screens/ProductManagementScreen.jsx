import React, { useState, useEffect } from 'react';
import { View, Text, Image, TouchableOpacity, ScrollView, TextInput, Modal, Alert, KeyboardAvoidingView } from 'react-native';
import { styles } from '../Styles/ProductManagementScreenStyle';
import TopNavigationPanel from '../../Navigation/Panels/Top/TopNavigationSortPanel';
import BottomNavigationPanel from '../../Navigation/Panels/Bottom/BottomNavigationWorkerPanel';
import axios from 'axios';
import LoadingModal from '../../Login/Components/LoadingModal';
import { DataTable } from 'react-native-paper';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import * as Keychain from 'react-native-keychain';

export default function ProductManagementScreen() {
    const [products, setProducts] = useState([]);
    const [sortOption, setSortOption] = useState('name');
    const [modalVisible, setModalVisible] = useState(false);
    const [selectedProduct, setSelectedProduct] = useState(null);
    const [newSize, setNewSize] = useState('');
    const [newQuantity, setNewQuantity] = useState('');
    const [loading, setLoading] = useState(true);
    const [modalType, setModalType] = useState('');
    const [showSort, setShowSort] = useState(false);

    useEffect(() => {
        fetchProducts();
    }, []);

    const fetchProducts = async () => {
        axios.get('https://sneakers-api.fly.dev/api/Product/products')
            .then(response => {
                setProducts(response.data); 
                setLoading(false);
            })
            .catch(error => {
                console.error('Error fetching products:', error);
                setLoading(false);
            });
    };

    const handleSort = (option) => {
        setSortOption(option);
        const sortedProducts = [...products];
        if (option === 'quantityDescending') {
            sortedProducts.sort((a, b) => 
                b.stocks.reduce((sum, size) => sum + size.quantity, 0) - 
                a.stocks.reduce((sum, size) => sum + size.quantity, 0)
            );
        } else if (option === 'quantityAscending') {
            sortedProducts.sort((a, b) => 
                a.stocks.reduce((sum, size) => sum + size.quantity, 0) -
                b.stocks.reduce((sum, size) => sum + size.quantity, 0)
            );
        } else if (option === 'name') {
            sortedProducts.sort((a, b) => a.model.localeCompare(b.model));
        }
        setProducts(sortedProducts);
        toggleSort();
    };

    const toggleSort = () => {
        setShowSort(!showSort);
    };

    const handleAddSize = async () => {
        const credentials = await Keychain.getGenericPassword();
        if (!credentials) {
            throw new Error('No credentials stored');
        }
        const parsedData = JSON.parse(credentials.password);
        const token = parsedData.token;

        try {
            const existingSize = selectedProduct.stocks.find(stock => stock.size === parseFloat(newSize));
            if (existingSize) {
                const updatedQuantity = existingSize.quantity + parseInt(newQuantity);
                await axios.put(`https://sneakers-api.fly.dev/api/Stock/${existingSize.id}`, {
                    quantity: updatedQuantity
                }, {
                    headers: { 
                        Authorization: `Bearer ${token}` 
                    }
                });
            } else {
                await axios.post(`https://sneakers-api.fly.dev/api/Stock/${selectedProduct.id}`, {
                    discount: 0,
                    quantity: parseInt(newQuantity),
                    size: parseFloat(newSize)
                }, {
                    headers: { 
                        Authorization: `Bearer ${token}`
                    }
                });
            }
            setModalVisible(false);
            fetchProducts();
        } catch (error) {
            console.error('Error adding/updating size:', error);
        }
    };

    const handleRemoveSize = async () => {
        const credentials = await Keychain.getGenericPassword();
        if (!credentials) {
            throw new Error('No credentials stored');
        }
        const parsedData = JSON.parse(credentials.password);
        const token = parsedData.token;

        const existingSize = selectedProduct.stocks.find(s => s.size === parseFloat(newSize));
        if (!existingSize) {
            Alert.alert('Error', 'This size does not exist.');
            return;
        }
        if (existingSize.quantity < parseInt(newQuantity)) {
            Alert.alert('Error', `You can only remove up to ${existingSize.quantity} items.`);
            return;
        }
        try {
            const updatedQuantity = existingSize.quantity - parseInt(newQuantity);
            if (updatedQuantity > 0) {
                await axios.put(`https://sneakers-api.fly.dev/api/Stock/${existingSize.id}`, {
                    quantity: updatedQuantity
                }, {
                    headers: { 
                        Authorization: `Bearer ${token}` 
                    }
                });
            } else {
                await axios.delete(`https://sneakers-api.fly.dev/api/Stock/${existingSize.id}`, {
                    headers: { 
                        Authorization: `Bearer ${token}` 
                    }
                });
            }
            setModalVisible(false);
            fetchProducts();
        } catch (error) {
            console.error('Error removing size:', error);
        }
    };

    const handleRemoveProduct = async (productId) => {
        const credentials = await Keychain.getGenericPassword();
        if (!credentials) {
            throw new Error('No credentials stored');
        }
        const parsedData = JSON.parse(credentials.password);
        const token = parsedData.token;
        
        Alert.alert(
            'Remove Product',
            'Are you sure you want to remove this product from the store?',
            [
                { text: 'No', style: 'cancel' },
                {
                    text: 'Yes', onPress: async () => {
                        try {
                            await axios.delete(`https://sneakers-api.fly.dev/api/Product/${productId}`, {
                                headers: { 
                                    Authorization: `Bearer ${token}` 
                                }
                            });
                            fetchProducts();
                        } catch (error) {
                            console.error('Error removing product:', error);
                        }
                    }
                }
            ]
        );
    };

    const renderProducts = () => {
        return products.map(product => (
            <Animated.View entering={FadeInUp.duration(1000).springify()} key={product.id} style={styles.product}>
                <View style={styles.productData}>
                    <Animated.View entering={FadeInUp.duration(1000).springify()} style={styles.productImageContainer}>
                        <Image style={styles.productImage} source={{ uri: product.photos[0].imgUrl }} />
                    </Animated.View>
                    <View style={styles.productInfoContainer}>
                        <Animated.Text entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.nameProductText}>{product.model}</Animated.Text>
                        <DataTable style={styles.sizeContainer}> 
                            <DataTable.Header> 
                                <DataTable.Title textStyle={styles.tableHeader} >Size</DataTable.Title> 
                                <DataTable.Title textStyle={styles.tableHeader} >Quantity</DataTable.Title> 
                            </DataTable.Header>
                            {product.stocks.map(stock => (
                                <DataTable.Row key={stock.id}> 
                                    <DataTable.Cell textStyle={styles.tableText} >{stock.size}</DataTable.Cell> 
                                    <DataTable.Cell textStyle={styles.tableText} >{stock.quantity}</DataTable.Cell> 
                                </DataTable.Row> 
                            ))} 
                        </DataTable>
                    </View>
                </View>

                <Animated.View entering={FadeInUp.duration(1000).springify().delay(500)} style={styles.priceContainer}>
                    <TouchableOpacity
                        style={styles.manageButtonContainer}
                        onPress={() => {
                            setSelectedProduct(product);
                            setModalType('add');
                            setModalVisible(true);
                        }}
                    >
                        <Text style={styles.manageButtonText}>Add Size</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                        style={styles.manageButtonContainer}
                        onPress={() => {
                            setSelectedProduct(product);
                            setModalType('remove');
                            setModalVisible(true);
                        }}
                    >
                        <Text style={styles.manageButtonText}>Remove Size</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                        style={styles.manageButtonContainer}
                        onPress={() => handleRemoveProduct(product.id)}
                    >
                        <Text style={styles.manageButtonText}>Remove Product</Text>
                    </TouchableOpacity>
                </Animated.View>
            </Animated.View>
        ));
    };

    const renderModalContent = () => (
        <Animated.View entering={FadeInDown.duration(1000).springify()} style={styles.modalContent}>
            <Animated.Text entering={FadeInDown.duration(1000).springify().randomDelay()} style={styles.modalTitle}>{modalType === 'add' ? 'Add New Size' : 'Remove Size'}</Animated.Text>
            <Animated.View entering={FadeInDown.duration(1000).springify().delay(100)}  style={{width: '100%'}}>
                <TextInput
                    style={styles.modalInput}
                    placeholder="Size"
                    value={newSize}
                    onChangeText={setNewSize}
                    keyboardType="decimal-pad"
                />
            </Animated.View>
            <Animated.View entering={FadeInDown.duration(1000).springify().delay(500)}  style={{width: '100%'}}>
                <TextInput
                    style={styles.modalInput}
                    placeholder="Quantity"
                    value={newQuantity}
                    onChangeText={setNewQuantity}
                    keyboardType="numeric"
                />
            </Animated.View>
            <Animated.View entering={FadeInDown.duration(1000).springify().delay(1000)}  style={{width: '100%'}}>
                <TouchableOpacity
                    style={styles.modalButton}
                    onPress={modalType === 'add' ? handleAddSize : handleRemoveSize}
                >
                    <Text style={styles.modalButtonText}>{modalType === 'add' ? 'Add' : 'Remove'}</Text>
                </TouchableOpacity>
            </Animated.View>
            <Animated.View entering={FadeInDown.duration(1000).springify().delay(1500)}  style={{width: '100%'}}>
                <TouchableOpacity
                    style={styles.modalButton}
                    onPress={() => setModalVisible(false)}
                >
                    <Text style={styles.modalButtonText}>Cancel</Text>
                </TouchableOpacity>
            </Animated.View>
        </Animated.View>
    );

    return (
        <View style={styles.body}>
            <TopNavigationPanel onPressSort={toggleSort} />
            <KeyboardAvoidingView behavior="height" style={styles.keyboardBox} keyboardVerticalOffset={25}>
                {showSort &&
                    <Animated.View entering={FadeInUp.duration(1000).springify()} style={styles.sortContent}>
                        <TouchableOpacity style={[styles.sortButton, sortOption === 'name' && styles.sortButtonChoosen]} onPress={() => handleSort('name')}>
                            <Animated.Text entering={FadeInUp.duration(1000).springify().delay(100)} style={[styles.sortButtonText, sortOption === 'name' && styles.sortButtonTextChoosen]}>Model</Animated.Text>
                        </TouchableOpacity>
                        <TouchableOpacity style={[styles.sortButton, sortOption === 'quantityAscending' && styles.sortButtonChoosen]} onPress={() => handleSort('quantityAscending')}>
                            <Animated.Text entering={FadeInUp.duration(1000).springify().delay(500)}  style={[styles.sortButtonText, sortOption === 'quantityAscending' && styles.sortButtonTextChoosen]}>Ilość rosnąco</Animated.Text>
                        </TouchableOpacity>
                        <TouchableOpacity style={[styles.sortButton, sortOption === 'quantityDescending' && styles.sortButtonChoosen]} onPress={() => handleSort('quantityDescending')}>
                            <Animated.Text entering={FadeInUp.duration(1000).springify().delay(1000)}  style={[styles.sortButtonText, sortOption === 'quantityDescending' && styles.sortButtonTextChoosen]}>Ilość malejąco</Animated.Text>
                        </TouchableOpacity>
                    </Animated.View>
                }
                <ScrollView contentContainerStyle={styles.mainContainer}>
                    {!loading &&
                        <View style={{ marginBottom: 20 }}>
                            {renderProducts()}
                        </View>
                    }
                </ScrollView>
            </KeyboardAvoidingView>
            <BottomNavigationPanel />
            <Modal
                animationType="slide"
                transparent={true}
                visible={modalVisible}
                onRequestClose={() => setModalVisible(false)}
            >
                <View style={styles.modalContainer}>
                    {renderModalContent()}
                </View>
            </Modal>
            <LoadingModal visible={loading} />
        </View>
    );
}
