import React, { useEffect, useState } from 'react';
import { View, Text, Image, TouchableOpacity, ScrollView, TextInput, KeyboardAvoidingView } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { styles } from '../Styles/MainMenuScreenStyle';
import TopNavigationPanel from '../../Navigation/Panels/Top/TopNavigationMenuPanel';
import BottomNavigationPanel from '../../Navigation/Panels/Bottom/BottomNavigationUserPanel';
import Ionicons from 'react-native-vector-icons/Ionicons';
import axios from 'axios';
import LoadingModal from '../../Login/Components/LoadingModal';
import Animated, { FadeIn, FadeInDown, FadeInUp } from 'react-native-reanimated';

export default function MainMenuScreen() {
    const [showFilter, setShowFilter] = useState(false);
    const [showSort, setShowSort] = useState(false);
    const [sortType, setSortType] = useState('default');
    const [products, setProducts] = useState([]);
    const [filteredProducts, setFilteredProducts] = useState([]);
    const [loading, setLoading] = useState(true);
        const [filters, setFilters] = useState({
        name: '',
        gender: {
            male: false,
            female: false,
            unisex: false,
        },
        priceRange: { min: '', max: '' },
        sizeRange: { min: '', max: '' },
    });

    useEffect(() => {
        axios.get('https://sneakers-api.fly.dev/api/Product/products')
            .then(response => {
                setProducts(response.data);
                setFilteredProducts(response.data);
                setLoading(false);
            })
            .catch(error => {
                console.error('Error fetching products:', error);
                setLoading(false);
            });
    }, []);

    const toggleFilter = () => {
        setShowFilter(!showFilter);
    };

    const toggleSort = () => {
        setShowSort(!showSort);
    };

    const applyFilters = () => {
        let filtered = [...products];

        if (filters.name) {
            filtered = filtered.filter(product => 
                product.model.toLowerCase().includes(filters.name.toLowerCase())
            );
        }

        const genders = Object.keys(filters.gender).filter(g => filters.gender[g]);
        if (genders.length > 0) {
            filtered = filtered.filter(product => 
                genders.includes(product.gender)
            );
        }

        if (filters.priceRange.min || filters.priceRange.max) {
            filtered = filtered.filter(product => {
                const price = product.price;
                return (!filters.priceRange.min || price >= parseFloat(filters.priceRange.min)) &&
                       (!filters.priceRange.max || price <= parseFloat(filters.priceRange.max));
            });
        }

        if (filters.sizeRange.min || filters.sizeRange.max) {
            filtered = filtered.filter(product => 
                product.stocks.some(stock => 
                    (!filters.sizeRange.min || stock.size >= parseFloat(filters.sizeRange.min)) &&
                    (!filters.sizeRange.max || stock.size <= parseFloat(filters.sizeRange.max))
                )
            );
        }

        setFilteredProducts(filtered);
    };

    const sortProducts = (type) => {
        let sortedProducts = [...filteredProducts];
        if (type === 'priceAscending') {
            sortedProducts.sort((a, b) => a.price - b.price);
        } else if (type === 'priceDescending') {
            sortedProducts.sort((a, b) => b.price - a.price);
        } else if (type === 'default') {
            sortedProducts.sort((a, b) => a.id.localeCompare(b.id));
        } else if (type === 'name') {
            sortedProducts.sort((a, b) => a.model.localeCompare(b.model));
        } else {
            sortedProducts.sort((a, b) => a.id.localeCompare(b.id));
        }
        setFilteredProducts(sortedProducts);
    };

    useEffect(() => {
        if (products.length > 0) {
            sortProducts(sortType);
        }
    }, [sortType]);
    
    useEffect(() => {
        applyFilters();
    }, [filters]);

    return (
        <View style={styles.body}>
            <TopNavigationPanel onPressFilter={toggleFilter} onPressSort={toggleSort} />
            <KeyboardAvoidingView behavior="height" style={styles.keyboardBox} keyboardVerticalOffset={20}>
            {showSort &&
                <SortContent setSortType={setSortType} sortType={sortType} toggleSort={toggleSort} />
            }
            {loading ? (
                <LoadingModal />
            ) : (
                <ScrollView contentContainerStyle={styles.mainContainer}>
                    {filteredProducts.map((product) => (
                        <ProductItem key={product.id} product={product} />
                    ))}
                </ScrollView>
            )}     

            </KeyboardAvoidingView>
            <BottomNavigationPanel />
            {showFilter && (
                <FilterSection filters={filters} setFilters={setFilters} toggleFilter={toggleFilter} />
            )}
        </View>
    );
}

function ProductItem({ product }) {
    const navigation = useNavigation();

    const handleProductPress = () => {
        navigation.navigate('Product', { productId: product.id });
    };

    return (
        <Animated.View entering={FadeInUp.duration(1000).springify().randomDelay()} >
        <TouchableOpacity style={styles.productContainer} onPress={handleProductPress}>
            <Animated.Image entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.productImage} source={{ uri: product.photos[0].imgUrl }} />
            <View style={styles.infoProductContainer}>
                <Animated.Text entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.infoProductText}>{product.model}</Animated.Text>
                <Animated.Text entering={FadeInUp.duration(1000).springify().randomDelay()}style={styles.infoProductText}>{product.price} zł</Animated.Text>
            </View>
        </TouchableOpacity>
        </Animated.View>
    );
}

function FilterSection({ filters, setFilters, toggleFilter }) {
    const handleInputChange = (name, value) => {
        setFilters(prevFilters => ({
            ...prevFilters,
            [name]: value,
        }));
    };

    const handleGenderChange = (gender) => {
        setFilters(prevFilters => ({
            ...prevFilters,
            gender: {
                ...prevFilters.gender,
                [gender]: !prevFilters.gender[gender]
            }
        }));
    };

    const handleRangeChange = (rangeName, type, value) => {
        setFilters(prevFilters => ({
            ...prevFilters,
            [rangeName]: {
                ...prevFilters[rangeName],
                [type]: value
            }
        }));
    };

    const resetFilters = () => {
        setFilters({
            name: '',
            gender: {
                male: false,
                female: false,
                unisex: false,
            },
            priceRange: { min: '', max: '' },
            sizeRange: { min: '', max: '' },
        });
    };

    return (
        <View style={styles.backgroundFilter}>
            <KeyboardAvoidingView behavior="paddding" style={styles.keyboardBox}  keyboardVerticalOffset={25}>
                <Animated.ScrollView entering={FadeIn.duration(1000).springify()}>
                    <View style={styles.filterContainer}>
                        <TouchableOpacity style={styles.closeImage} onPress={toggleFilter}>
                            <Image style={styles.closeImage} source={require('../../../../assets/Images/Menu/CloseIcon.png')} />
                        </TouchableOpacity>

                        <ScrollView contentContainerStyle={styles.filterContentContainer}>
                            <View style={styles.inputSection}>
                                <Animated.Text entering={FadeInDown.duration(1000).springify()} style={styles.titleText}>Nazwa</Animated.Text>
                                <Animated.View entering={FadeInDown.duration(1000).springify()} style={styles.inputName}>
                                    <TextInput
                                        style={styles.inputText}
                                        value={filters.name}
                                        onChangeText={(text) => handleInputChange('name', text)}
                                    />
                                </Animated.View>
                            </View>

                            <View style={styles.borderFilder}></View>

                            <View style={styles.checkboxContainer}>
                                <Animated.Text entering={FadeInDown.duration(1000).springify()} style={styles.titleText}>Płeć</Animated.Text>
                                <TouchableOpacity
                                    style={[styles.checkbox, filters.gender.male && styles.checked]}
                                    onPress={() => handleGenderChange('male')}>
                                    <Animated.View entering={FadeInDown.duration(1000).springify()} style={styles.checkboxIcon}>
                                        {filters.gender.male && <Ionicons name={'checkmark-outline'} size={20} color={'#411c5d'} style={styles.IconsSize} />}
                                    </Animated.View>
                                    <Animated.Text entering={FadeInDown.duration(1000).springify()} style={styles.checkboxText}>Mężczyźni</Animated.Text>
                                </TouchableOpacity>
                                <TouchableOpacity
                                    style={[styles.checkbox, filters.gender.female && styles.checked]}
                                    onPress={() => handleGenderChange('female')}>
                                    <Animated.View entering={FadeInDown.duration(1000).springify()} style={styles.checkboxIcon}>
                                        {filters.gender.female && <Ionicons name={'checkmark-outline'} size={20} color={'#411c5d'} style={styles.IconsSize} />}
                                    </Animated.View>
                                    <Animated.Text entering={FadeInDown.duration(1000).springify()} style={styles.checkboxText}>Kobiety</Animated.Text>
                                </TouchableOpacity>
                                <TouchableOpacity
                                    style={[styles.checkbox, filters.gender.unisex && styles.checked]}
                                    onPress={() => handleGenderChange('unisex')}>
                                    <Animated.View entering={FadeInDown.duration(1000).springify()} style={styles.checkboxIcon}>
                                        {filters.gender.unisex && <Ionicons name={'checkmark-outline'} size={20} color={'#411c5d'} style={styles.IconsSize} />}
                                    </Animated.View>
                                    <Animated.Text entering={FadeInDown.duration(1000).springify()} style={styles.checkboxText}>Uniseks</Animated.Text>
                                </TouchableOpacity>
                            </View>

                            <View style={styles.borderFilder}></View>

                            <View style={styles.inputSection}>
                                <Animated.Text entering={FadeInDown.duration(1000).springify()} style={styles.titleText}>Cena</Animated.Text>
                                <View style={styles.inputContainers}>
                                    <Animated.View entering={FadeInDown.duration(1000).springify()} style={styles.input}>
                                        <TextInput
                                            style={styles.inputText}
                                            placeholder="od"
                                            keyboardType="numeric"
                                            value={filters.priceRange.min}
                                            onChangeText={(text) => handleRangeChange('priceRange', 'min', text)}
                                        />
                                    </Animated.View>
                                        <Animated.View entering={FadeInDown.duration(1000).springify()} style={styles.input}>
                                        <TextInput
                                            style={styles.inputText}
                                            placeholder="do"
                                            keyboardType="numeric"
                                            value={filters.priceRange.max}
                                            onChangeText={(text) => handleRangeChange('priceRange', 'max', text)}
                                        />
                                    </Animated.View>
                                </View>
                            </View>

                            <View style={styles.borderFilder}></View>

                            <View style={styles.inputSection}>
                                <Animated.Text entering={FadeInDown.duration(1000).springify()} style={styles.titleText}>Rozmiar</Animated.Text>
                                <View style={styles.inputContainers}>
                                    <Animated.View entering={FadeInDown.duration(1000).springify()} style={styles.input}>
                                        <TextInput
                                            style={styles.inputText}
                                            placeholder="od"
                                            keyboardType="numeric"
                                            value={filters.sizeRange.min}
                                            onChangeText={(text) => handleRangeChange('sizeRange', 'min', text)}
                                        />
                                    </Animated.View>
                                    <Animated.View entering={FadeInDown.duration(1000).springify()} style={styles.input}>
                                        <TextInput
                                            style={styles.inputText}
                                            placeholder="do"
                                            keyboardType="numeric"
                                            value={filters.sizeRange.max}
                                            onChangeText={(text) => handleRangeChange('sizeRange', 'max', text)}
                                        />
                                    </Animated.View>
                                </View>
                            </View>
                        </ScrollView>
                        <Animated.View entering={FadeInDown.duration(1000).springify()} style={styles.buttonFilterContainer}>
                            <TouchableOpacity style={styles.buttonFilter} onPress={resetFilters}>
                                <Text style={styles.textButtonFilter}>Wyczyść</Text>
                            </TouchableOpacity>
                            <TouchableOpacity style={styles.buttonFilter} onPress={toggleFilter}>
                                <Text style={styles.textButtonFilter}>Zatwierdź</Text>
                            </TouchableOpacity>
                        </Animated.View>

                    </View>
                </Animated.ScrollView>
            </KeyboardAvoidingView>
        </View>
    );
}


function SortContent({ setSortType, sortType, toggleSort }) {
    return (
        <Animated.View entering={FadeInUp.duration(500).springify()} style={styles.sortContent}>
            <Animated.Text entering={FadeInUp.duration(1000).springify()} style={styles.sortTitle}>Sortuj według</Animated.Text>
            <TouchableOpacity style={[styles.sortButton, sortType === 'priceAscending' && styles.sortButtonChoosen]} onPress={async () => { await setSortType('priceAscending'); toggleSort() }}>
                <Animated.Text entering={FadeInUp.duration(1000).springify()} style={[styles.sortButtonText, sortType === 'priceAscending' && styles.sortButtonTextChoosen]}>Cena rosnąco</Animated.Text>
                {sortType === 'priceAscending' && <Ionicons name={'checkmark-outline'} size={30} color={'#411c5d'} style={styles.IconsSize} />}
            </TouchableOpacity>
            <TouchableOpacity style={[styles.sortButton, sortType === 'priceDescending' && styles.sortButtonChoosen]} onPress={async () => { await setSortType('priceDescending'); toggleSort() }}>
            <   Animated.Text entering={FadeInUp.duration(1000).springify()} style={[styles.sortButtonText, sortType === 'priceDescending' && styles.sortButtonTextChoosen]}>Cena malejąco</Animated.Text>
                {sortType === 'priceDescending' && <Ionicons name={'checkmark-outline'} size={30} color={'#411c5d'} style={styles.IconsSize} />}
            </TouchableOpacity>
            <TouchableOpacity style={[styles.sortButton, sortType === 'name' && styles.sortButtonChoosen]} onPress={async () => { await setSortType('name'); toggleSort() }}>
                <Animated.Text entering={FadeInUp.duration(1000).springify()} style={[styles.sortButtonText, sortType === 'name' && styles.sortButtonTextChoosen]}>Model</Animated.Text>
                {sortType === 'name' && <Ionicons name={'checkmark-outline'} size={30} color={'#411c5d'} style={styles.IconsSize} />}
            </TouchableOpacity>
            <TouchableOpacity style={[styles.sortButton, sortType === 'default' && styles.sortButtonChoosen]} onPress={async () => { await setSortType('default'); toggleSort() }}>
                <Animated.Text entering={FadeInUp.duration(1000).springify()} style={[styles.sortButtonText, sortType === 'default' && styles.sortButtonTextChoosen]}>Domyślnie</Animated.Text>
                {sortType === 'default' && <Ionicons name={'checkmark-outline'} size={30} color={'#411c5d'} style={styles.IconsSize} />}
            </TouchableOpacity>
        </Animated.View>
    );
}
